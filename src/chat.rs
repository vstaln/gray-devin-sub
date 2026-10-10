//! Chat turn over the loopback relay: the OpenAI Responses body the host
//! POSTs is answered by a pooled `devin acp` session (see [`crate::session`])
//! whose native tools are fully denied.
//!
//! The funnel contract (verified against `devin acp` behaviour):
//! * ACP has no system role and no assistant replay: the funnel system text
//!   (contract + tool manifest + host instructions) leads the first prompt,
//!   then the labeled transcript. Devin's prompt cache is per-session, so a
//!   request whose history strictly extends what a pooled session last
//!   answered prompts that session with only the delta — the transcript is
//!   already upstream. One upstream request per turn either way.
//! * gray tools are NOT native tools. They are described in the system text
//!   and the model answers with a fenced ```gray_calls block — gated by the
//!   `/devin tools` allowlist (default bash-only, [`crate::settings`]).
//!   Models that
//!   ignore the text funnel and reach for a native tool anyway are caught by
//!   `tool_call` notifications: execute/read redirect onto `bash`, anything
//!   else echoes back `native__`-prefixed so the host's unknown-tool error
//!   lists the real surface instead of silently ending the turn (see
//!   [`redirect_call`], [`blocked_call`]). A project-level `.devin/config.json`
//!   deny list makes every native tool fail closed regardless.
//! * `session/request_permission` is always answered `cancelled`: nothing a
//!   harness turn does may wait on a prompt.
//! * The relay speaks the OpenAI Responses SSE wire the host already streams,
//!   so no host changes are needed: the declared transport points at the
//!   per-turn relay URL and the host POSTs its standard body with the
//!   per-turn bearer.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::session;

/// Relay rejection when native retries past the single admitted request.
pub const ADMISSION_CONSUMED: &str = "DEVIN_MODEL_ADMISSION_CONSUMED";

/// One translated turn: funnel system text, transcript, host tool names.
pub struct PreparedTurn {
    pub system: String,
    pub content_line: String,
    /// The request's raw `input` array: continuation matching keys off it.
    pub input: Vec<Value>,
    pub names: Vec<String>,
    pub native_model: String,
}

/// Normalize a tool input schema: strip top-level `oneOf`/`allOf`/`anyOf`
/// and guarantee object schemas carry `properties`.
pub fn normalize_input_schema(schema: &Value) -> Value {
    let mut out = schema.clone();
    if let Some(obj) = out.as_object_mut() {
        for key in ["oneOf", "allOf", "anyOf"] {
            obj.remove(key);
        }
        obj.entry("type".to_string())
            .or_insert(Value::String("object".to_string()));
        if obj.get("type").and_then(Value::as_str) == Some("object")
            && !matches!(obj.get("properties"), Some(Value::Object(_)))
        {
            obj.insert("properties".to_string(), json!({}));
        }
    }
    out
}

fn check_tool_name(name: &str, seen: &std::collections::HashSet<String>) -> Result<(), String> {
    if name.len() > 50
        || name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        || seen.contains(name)
    {
        return Err(format!(
            "tool names must be unique ASCII identifiers of at most 50 characters: {name:?}"
        ));
    }
    Ok(())
}

fn text_of(blocks: &Value) -> String {
    match blocks {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => b.get("text").and_then(Value::as_str).map(str::to_string),
                // OpenAI Responses wire parts the host actually sends.
                Some("input_text" | "output_text") => {
                    b.get("text").and_then(Value::as_str).map(str::to_string)
                }
                // Inline images travel as ACP image blocks after the prompt
                // text (`images_of`); the transcript keeps a marker.
                Some("input_image") if image_block(b).is_some() => {
                    Some("[image attached]".to_string())
                }
                Some("input_image" | "image") => Some("[image omitted]".to_string()),
                Some("input_file" | "input_video") => Some("[file omitted]".to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// ACP image block for a Responses `input_image` part carrying a
/// `data:<mime>;base64,<data>` URL. Remote URLs aren't fetched.
fn image_block(part: &Value) -> Option<Value> {
    let url = part.get("image_url")?;
    let url = url.as_str().or_else(|| url.get("url")?.as_str())?;
    let (mime, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
    Some(json!({"type": "image", "mimeType": mime, "data": data}))
}

/// ACP image blocks for every inline image in `items`' messages, in order.
pub(crate) fn images_of(items: &[Value]) -> Vec<Value> {
    items
        .iter()
        .filter(|i| item_kind(i) == "message")
        .filter_map(|i| i.get("content")?.as_array())
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("input_image"))
        .filter_map(image_block)
        .collect()
}

/// Kind of a Responses input item: the `type` field, or "message" for the
/// EasyInputMessage short form (role present, type absent) the host emits.
fn item_kind(item: &Value) -> &str {
    match item.get("type").and_then(Value::as_str) {
        Some(k) => k,
        None if item.get("role").is_some() => "message",
        None => "",
    }
}

/// Render one Responses `input` item to transcript text. Tool calls and
/// results render as explicit records so the model can reference them;
/// reasoning carriers restore nothing on a fresh session but their text
/// still carries the prior answer.
fn render_item(item: &Value) -> Option<String> {
    let kind = item_kind(item);
    match kind {
        "message" => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
            let text = text_of(item.get("content").unwrap_or(&Value::Null));
            if text.trim().is_empty() {
                return None;
            }
            let who = match role {
                "assistant" => "Assistant",
                "system" | "developer" => "System",
                _ => "User",
            };
            Some(format!("[{who}]\n{text}"))
        }
        "function_call" => {
            let name = item.get("name").and_then(Value::as_str).unwrap_or("?");
            let args = item.get("arguments").and_then(Value::as_str).unwrap_or("");
            let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
            Some(format!("[Tool call {name} id={call_id}]\n{args}"))
        }
        "function_call_output" => {
            let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
            let out = item
                .get("output")
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| v.to_string())
                })
                .unwrap_or_default();
            Some(format!("[Tool result id={call_id}]\n{out}"))
        }
        "reasoning" => {
            let text = item
                .get("summary")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            if text.trim().is_empty() {
                None
            } else {
                Some(format!("[Assistant reasoning]\n{text}"))
            }
        }
        _ => None,
    }
}

/// Render a Responses `input` array to transcript text.
fn render_items(items: &[Value]) -> String {
    items
        .iter()
        .filter_map(render_item)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Translate an OpenAI Responses body into one funnel turn.
pub fn prepare_turn(body: &Value, model: &str) -> Result<PreparedTurn, String> {
    let instructions = body
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let input = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut names: Vec<String> = Vec::new();
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut tool_specs: Vec<String> = Vec::new();
    // The operator's allowlist (`/devin tools`, default bash-only): only
    // passing tools are described upstream or callable through the funnel.
    let policy = crate::settings::ToolPolicy::load();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for t in tools {
            let name = t.get("name").and_then(Value::as_str).unwrap_or("");
            // Responses tools are functions; anything else is host-owned.
            if t.get("type")
                .and_then(Value::as_str)
                .is_some_and(|k| k != "function")
            {
                continue;
            }
            // Filter before validating: a disallowed tool is invisible
            // here, so its name (valid or not) can never fail a turn.
            if !policy.allows(name) {
                continue;
            }
            check_tool_name(name, &seen_names)?;
            seen_names.insert(name.to_string());
            names.push(name.to_string());
            let desc = t.get("description").and_then(Value::as_str).unwrap_or("");
            let params = normalize_input_schema(t.get("parameters").unwrap_or(&json!({})));
            tool_specs.push(format!("- {name}: {desc} arguments schema: {params}"));
        }
    }
    if input.is_empty() {
        return Err("history must end in a nonempty user/tool-result message".into());
    }
    // The transcript must end in something the model should answer: a
    // user/assistant message, a tool call, or a tool result. A trailing
    // reasoning item alone answers nothing.
    let last_kind = input.last().map(item_kind).unwrap_or("");
    if !matches!(
        last_kind,
        "message" | "function_call" | "function_call_output"
    ) {
        return Err(
            "history must end in a nonempty user/tool-result message; assistant prefill is unsupported"
                .into(),
        );
    }
    let content_line = render_items(&input);
    if content_line.is_empty() {
        return Err("history must end in a nonempty user/tool-result message".into());
    }
    // Assistant prefill (trailing assistant message, no tool result after
    // it) is unsupported: the funnel answers the transcript as-is.
    if input
        .last()
        .and_then(|i| i.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
    {
        return Err(
            "history must end in a nonempty user/tool-result message; assistant prefill is unsupported"
                .into(),
        );
    }
    let mut system_parts: Vec<String> = Vec::new();
    if !instructions.is_empty() {
        system_parts.push(instructions.to_string());
    }
    system_parts.push(funnel_contract(&tool_specs));
    // Responses `reasoning.effort` is the host's /thinking pick; the model
    // id family + tier name resolve to the real native id (the catalog
    // collapses `swe-2-high|medium|max` into one `swe-2` row, so the effort
    // is how a tier actually gets chosen).
    let effort = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str);
    Ok(PreparedTurn {
        system: system_parts.join("\n\n"),
        content_line,
        input,
        names,
        native_model: crate::catalog::native_model(model, effort),
    })
}

/// The text funnel contract the model answers with. Verified with Claude
/// Sonnet: it emits the block exactly as specified.
fn funnel_contract(tool_specs: &[String]) -> String {
    let mut s = String::from(
        "You are the model behind another agent harness (\"gray\"). Your own built-in tools are \
        DISABLED in this session: every call to them is denied and wastes the turn. Never call them.\n\
        Instead, the harness executes tools for you. To call harness tools, end your reply with \
        exactly one fenced block:\n\n```gray_calls\n\
        [{\"name\": \"<tool name>\", \"arguments\": { ... }}]\n```\n\n\
        The array may hold several calls. Do not write anything after the block. \
        If no tool is needed, answer normally with no block. \
        Never call a tool just to wait, poll or pass a round (`true`, `:`, `sleep`, `echo`): \
        to wait for a background job use its tool's own wait (bash `action: \"output\"` with \
        `job_id` and `wait_ms`), otherwise reply with no block. \
        Later messages in this conversation come from the harness: tool results \
        arrive as `[Tool result id=…]` blocks and new user input as `[User]` blocks.",
    );
    s.push_str("\n\nHarness tools:\n");
    if tool_specs.is_empty() {
        s.push_str("none for this request.");
    } else {
        s.push_str(&tool_specs.join("\n"));
    }
    s
}

/// Parse the funnel: the LAST real ```gray_calls block in `text` — real
/// meaning a bare info string (the next char is a newline or EOF) and a
/// body that leads with `[`. Inline mentions and fenced prose stay in
/// the answer text, where the host's leaked-markup nudge can answer
/// them. No real block → `None` (plain answer). A real block is always
/// the model asking for tools, so it is never dumped as text (that ends
/// the turn on raw JSON): non-object arguments pass through as-is, an
/// unparseable array becomes one call carrying the raw block, and a
/// name outside the advertised set folds to `native__…` — the host's
/// error result names the real surface either way, so the model sees
/// why and retries.
///
/// The closing fence is optional: SWE models often terminate the call
/// list with their native pipe-delimited tool markup instead of ``` — or
/// emit no close at all. In both cases the JSON array still leads the
/// tail, so a prefix parse recovers it and the markup is dropped.
pub fn parse_calls_block(text: &str, names: &[String]) -> Option<(String, Vec<(String, String)>)> {
    const FENCE: &str = "```gray_calls";
    // Scan occurrences last-to-first; the first real block wins. A real
    // block's info string ends at the newline (or EOF) and its body leads
    // with `[` — an inline mention (`the <fence> contract)`) or a fenced
    // explanation is prose about the funnel, not a call, so it stays in
    // the text where the host's leaked-markup nudge can answer it.
    let mut search_from = text.len();
    let (start, candidate) = loop {
        let idx = text[..search_from].rfind(FENCE)?;
        let after = &text[idx + FENCE.len()..];
        let info_ends = after.is_empty() || after.starts_with('\n');
        let body = match after.find("```") {
            Some(close) => &after[..close],
            None => after,
        }
        .trim();
        if info_ends && body.starts_with('[') {
            break (idx, body);
        }
        search_from = idx;
    };
    // The tool the model most likely meant, so the host's error names it.
    let guess = names
        .iter()
        .find(|n| {
            candidate.contains(&format!("\"name\":\"{n}\""))
                || candidate.contains(&format!("\"name\": \"{n}\""))
        })
        .or(names.first())
        .map_or("gray_calls", String::as_str);
    let calls: Vec<(String, String)> = match json_array_prefix(candidate) {
        Some(arr) => arr
            .iter()
            .map(|c| {
                let name = c.get("name").and_then(Value::as_str).unwrap_or(guess);
                let args = c.get("arguments").unwrap_or(&Value::Null);
                // The advertised set is the whole permit: a name outside
                // `names` can't run even when the host owns such a tool —
                // fold to `native__…` so the "does not exist" error
                // teaches the real surface instead of silently executing.
                let name = if names.iter().any(|n| n == name) {
                    name.to_string()
                } else {
                    native_name(name)
                };
                (name, args.to_string())
            })
            .collect(),
        None => vec![(guess.to_string(), candidate.to_string())],
    };
    Some((text[..start].trim_end().to_string(), calls))
}

/// A leading self-delimiting JSON array, ignoring whatever follows: the
/// model may append native tool markup or hallucinated transcript turns
/// after the array, and both are noise to drop, not parse failures.
fn json_array_prefix(s: &str) -> Option<Vec<Value>> {
    serde_json::Deserializer::from_str(s)
        .into_iter::<Vec<Value>>()
        .next()?
        .ok()
}

/// Redirect a native `tool_call` notification onto a host tool — `bash` for
/// execute/read/edit/write, the host's own web tools for search/fetch —
/// when the tool exists in `names`. Verified shapes: `{"kind":"execute",
/// "rawInput":{"command": "..."}}` and `{"kind":"read","rawInput":
/// {"file_path":"..."}}`.
pub(crate) fn redirect_call(update: &Value, names: &[String]) -> Option<(String, String)> {
    let kind = update.get("kind").and_then(Value::as_str).unwrap_or("");
    let input = update.get("rawInput").unwrap_or(&Value::Null);
    if let Some(call) = mcp_redirect(input, names) {
        return Some(call);
    }
    let has = |n: &str| names.iter().any(|x| x == n);
    let bash = |cmd: String| {
        has("bash").then(|| ("bash".to_string(), json!({"command": cmd}).to_string()))
    };
    match kind {
        "execute" => input
            .get("command")
            .and_then(Value::as_str)
            .and_then(|c| bash(c.to_string())),
        "read" => input
            .get("file_path")
            .and_then(Value::as_str)
            .and_then(|p| bash(format!("cat -- {}", shell_quote(p)))),
        // edit/write land on a python3 one-liner instead of the native__ dead
        // end: same file semantics, still host-mediated through `bash`.
        "edit" => edit_redirect(input).and_then(|a| bash_args(a, has("bash"))),
        "write" | "create" => write_redirect(input).and_then(|a| bash_args(a, has("bash"))),
        // Devin reports BOTH web search and web fetch as `kind: "fetch"`
        // (verified: search carries `rawInput.query` and
        // `_meta["cognition.ai/inferenceToolName"] = "web_search"`, fetch
        // carries `rawInput.url` and `"webfetch"`), so classify by payload,
        // not kind.
        "search" | "fetch" => web_redirect(update, input, names),
        _ => None,
    }
}

/// Devin's `mcp_call_tool` aimed at a harness tool. The prompt calls our
/// surface "harness tools", so the model sometimes reaches for them via its
/// native MCP bridge: `{"server_name":"harness","tool_name":"bash",
/// "arguments":{…}}` (titled "Calling bash from harness"). When `tool_name`
/// is on the allowed surface, unwrap it into the call it meant; web tools
/// go through [`web_redirect`] so a disallowed fetch still lands on curl.
/// Anything else (a real MCP server's tool) stays blocked.
fn mcp_redirect(input: &Value, names: &[String]) -> Option<(String, String)> {
    let tool = input.get("tool_name").and_then(Value::as_str)?;
    input.get("server_name")?;
    let args = match input.get("arguments") {
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or(Value::Null),
        Some(v) => v.clone(),
        None => Value::Null,
    };
    let args = if args.is_object() { args } else { json!({}) };
    if matches!(tool, "web_fetch" | "web_search" | "webfetch") {
        let meta = if tool == "web_search" {
            "web_search"
        } else {
            "webfetch"
        };
        let update = json!({"_meta": {"cognition.ai/inferenceToolName": meta}});
        return web_redirect(&update, &args, names);
    }
    names
        .iter()
        .any(|n| n == tool)
        .then(|| (tool.to_string(), args.to_string()))
}

/// Native web search/fetch → the host's own tools when the policy allows
/// them. A fetch the policy keeps off `web_fetch` still lands on `bash` as
/// `curl` when bash is allowed — same capability the bash surface already
/// grants, and what the model falls back to by hand otherwise. A search
/// with no `web_search` has no faithful bash equivalent: it folds to a
/// self-explaining `native__` name so the host error says why.
fn web_redirect(update: &Value, input: &Value, names: &[String]) -> Option<(String, String)> {
    let has = |n: &str| names.iter().any(|x| x == n);
    let tool = update
        .pointer("/_meta/cognition.ai~1inferenceToolName")
        .and_then(Value::as_str)
        .unwrap_or("");
    let url = input.get("url").and_then(Value::as_str);
    let query = input.get("query").and_then(Value::as_str);
    let is_search = tool == "web_search" || (url.is_none() && query.is_some());
    if is_search {
        let q = query?;
        return Some(if has("web_search") {
            ("web_search".to_string(), json!({"query": q}).to_string())
        } else {
            (
                native_name("web_search_not_enabled_for_devin"),
                "{}".to_string(),
            )
        });
    }
    let url = url?;
    if has("web_fetch") {
        return Some(("web_fetch".to_string(), json!({"url": url}).to_string()));
    }
    has("bash").then(|| {
        let cmd = format!(
            "curl -fsSL --max-time 30 -A 'Mozilla/5.0' -- {} | head -c 200000",
            shell_quote(url)
        );
        ("bash".to_string(), json!({"command": cmd}).to_string())
    })
}

fn bash_args(args_json: String, has_bash: bool) -> Option<(String, String)> {
    has_bash.then(|| ("bash".to_string(), args_json))
}

/// `edit` → exact-string replace via python3. `old_string` must appear
/// exactly once unless `replace_all` is set — same contract as the native
/// tool, so the error text still teaches the model what went wrong.
fn edit_redirect(input: &Value) -> Option<String> {
    let path = input.get("file_path").and_then(Value::as_str)?;
    let old_s = input.get("old_string").and_then(Value::as_str)?;
    let new_s = input
        .get("new_string")
        .and_then(Value::as_str)
        .unwrap_or("");
    let replace_all = input
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let payload = json!({
        "file_path": path, "old_string": old_s,
        "new_string": new_s, "replace_all": replace_all,
    })
    .to_string();
    let script = "import json,sys
        a=json.loads(sys.argv[1]);p=a['file_path']
        s=open(p,encoding='utf-8').read()
        o,n=a['old_string'],a['new_string']
        if a.get('replace_all'): out=s.replace(o,n)
        else:
            c=s.count(o)
            assert c==1,f'old_string found {c} times in {p} (need exactly 1)'
            out=s.replace(o,n,1)
        open(p,'w',encoding='utf-8').write(out)";
    Some(
        json!({"command": format!("python3 -c {} {}", shell_quote(script), shell_quote(&payload))})
            .to_string(),
    )
}

/// `write`/`create` → python3 writes the content verbatim.
fn write_redirect(input: &Value) -> Option<String> {
    let path = input.get("file_path").and_then(Value::as_str)?;
    let content = input.get("content").and_then(Value::as_str).unwrap_or("");
    let payload = json!({"file_path": path, "content": content}).to_string();
    let script = "import json,sys
        a=json.loads(sys.argv[1])
        open(a['file_path'],'w',encoding='utf-8').write(a['content'])";
    Some(
        json!({"command": format!("python3 -c {} {}", shell_quote(script), shell_quote(&payload))})
            .to_string(),
    )
}

/// The call a native `tool_call` becomes when [`redirect_call`] can't map
/// it onto bash (edit/fetch/other kinds, or a bash-less surface): the name
/// the model reached for — the update's `title`, else `kind` — sanitized
/// and `native__`-prefixed so it can never collide with a real tool. The
/// host answers "Tool 'native__X' does not exist. Available: …", an error
/// result that names the true surface and keeps the turn alive; a dropped
/// native call is how turns end dead text-only.
pub(crate) fn blocked_call(update: &Value) -> (String, String) {
    let raw = update
        .get("title")
        .and_then(Value::as_str)
        .or_else(|| update.get("kind").and_then(Value::as_str))
        .unwrap_or("tool_call");
    (native_name(raw), "{}".to_string())
}

/// `native__`-prefixed form of a name the model reached for — sanitized,
/// capped at 50 chars, and unable to collide with a real tool. Shared by
/// [`blocked_call`] (native tool calls) and [`parse_calls_block`] (funnel
/// calls outside the advertised set): both end at the host's
/// "does not exist" error.
pub(crate) fn native_name(raw: &str) -> String {
    let mut name = String::from("native__");
    for c in raw.chars() {
        if name.len() >= 50 {
            break;
        }
        name.push(if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            c
        } else {
            '_'
        });
    }
    name
}

/// POSIX single-quote a path: `'` becomes `'\''`.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// An input item the assistant side produced: the replayed echo of a
/// session's own answer (assistant message, its calls, reasoning carriers).
fn assistant_side(item: &Value) -> bool {
    match item_kind(item) {
        "function_call" | "reasoning" => true,
        "message" => item.get("role").and_then(Value::as_str) == Some("assistant"),
        _ => false,
    }
}

/// Prepended to a continuation delta whose previous reply never reached
/// the host.
pub(crate) const UNDELIVERED_NOTE: &str = "[Harness note] Your previous reply was interrupted by the user and never delivered; none of its tool calls ran.";

/// Strict-continuation check: `input` is `absorbed` plus the echo of the
/// session's own last answer plus a non-assistant tail. Returns the tail
/// rendered to transcript text — the only thing the session still needs
/// to see. Miss reasons feed the DEVIN_SUB_DEBUG trace: `prefix` (history
/// diverged), `echo` (the replayed answer isn't what this session sent),
/// `empty_delta` (nothing new to ask, or an assistant item sits in the
/// new tail, which means the history mid-edited a turn).
///
/// The host echoes our answer as `{"role":"assistant","content":<text>}`
/// (only when the text is non-empty) then one `{"type":"function_call",...}`
/// item per call; reasoning items may interleave and carry nothing here.
pub(crate) fn continuation(
    absorbed: &[Value],
    reply_call_ids: &[String],
    reply_text: &str,
    input: &[Value],
) -> Result<String, &'static str> {
    if input.len() <= absorbed.len() || !input.starts_with(absorbed) {
        return Err("prefix");
    }
    let rest = &input[absorbed.len()..];
    let mut i = 0;
    let mut call_ids: Vec<String> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    while i < rest.len() && assistant_side(&rest[i]) {
        let item = &rest[i];
        match item_kind(item) {
            "function_call" => call_ids.push(
                item.get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            "message" => texts.push(text_of(item.get("content").unwrap_or(&Value::Null))),
            _ => {}
        }
        i += 1;
    }
    // An empty echo zone against a non-empty recorded reply: the reply
    // never reached gray (interrupted turn, client gone before delivery).
    // The child still holds it, so continue with a note instead of
    // re-billing the whole prefix on a fresh session.
    let undelivered = i == 0 && (!reply_call_ids.is_empty() || !reply_text.trim().is_empty());
    if !undelivered
        && (call_ids.as_slice() != reply_call_ids || texts.join("\n").trim() != reply_text.trim())
    {
        return Err("echo");
    }
    let tail = &rest[i..];
    if tail.is_empty() || tail.iter().any(assistant_side) {
        return Err("empty_delta");
    }
    let delta = render_items(tail);
    if delta.trim().is_empty() {
        return Err("empty_delta");
    }
    if undelivered {
        return Ok(format!("{UNDELIVERED_NOTE}\n\n{delta}"));
    }
    Ok(delta)
}

/// Usage snapshot from a `usage_update` notification's `update` object or
/// a prompt result's `usage` value: plain key first, then Cognition's flat
/// `_meta["cognition.ai/<key>"]`, then the nested `_meta["cognition.ai"]
/// [<key>]` object. `None` when every count is zero or absent.
pub fn usage_from(v: &Value) -> Option<Usage> {
    let get = |k: &str| {
        let flat = format!("cognition.ai/{k}");
        v.get(k)
            .or_else(|| v.get("_meta").and_then(|m| m.get(flat.as_str())))
            .or_else(|| {
                v.get("_meta")
                    .and_then(|m| m.get("cognition.ai"))
                    .and_then(|c| c.get(k))
            })
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize
    };
    let usage = Usage {
        input_tokens: get("inputTokens"),
        output_tokens: get("outputTokens"),
        cached_tokens: get("cachedReadTokens"),
        cache_write_tokens: get("cachedWriteTokens"),
    };
    if usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_tokens == 0
        && usage.cache_write_tokens == 0
    {
        None
    } else {
        Some(usage)
    }
}

/// What one `devin acp` session produced.
pub struct TurnResult {
    pub text: String,
    /// Accumulated agent_thought_chunk text (emitted as reasoning summary).
    pub thought: String,
    /// (call_id, name, arguments-json) in emission order.
    pub calls: Vec<(String, String, String)>,
    pub usage: Usage,
    pub stop: String,
    /// The relay client was already gone when this turn settled: the
    /// reply was produced but never delivered, so the session records an
    /// empty echo for it (see `LiveSession::absorb`).
    pub unseen: bool,
}

#[derive(Default, Clone)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_tokens: usize,
    /// Prompt tokens written to the cache (`cachedWriteTokens`); like
    /// `cached_tokens`, already counted inside `input_tokens`.
    pub cache_write_tokens: usize,
}

/// Run one turn: prompt a pooled continuation session with only the delta
/// when this request's history strictly extends what it last answered,
/// else spawn `devin acp`, drive the ACP handshake and prompt with
/// `system` + the whole transcript. `timeout` is the whole-turn budget;
/// `cancel` is set when the relay client went away mid-turn. A failure
/// whose prompt still settled upstream returns the session to the pool
/// (`prompt_in_flight` cleared): the host retries a failed turn with an
/// identical request, and a pooled continuation session re-answers it
/// with just the delta instead of re-billing the transcript.
pub fn run_turn(
    turn: &PreparedTurn,
    cancel: &Arc<AtomicBool>,
    timeout: Duration,
) -> Result<TurnResult, String> {
    let deadline = Instant::now() + timeout;
    let prompt_text = if turn.system.is_empty() {
        turn.content_line.clone()
    } else {
        format!("{}\n\n{}", turn.system, turn.content_line)
    };
    let chars = prompt_text.chars().count();
    let (claimed, mut reason) = session::take(turn, cancel);
    let spawn_fresh = |reason: &str| match session::spawn(turn, deadline, cancel) {
        Ok(s) => Ok(s),
        Err(e) => {
            trace_turn(&format!("fresh:{reason}"), "-", chars, None, Some(&e));
            Err(e)
        }
    };
    let mut s = match claimed {
        Some((mut s, delta)) => {
            let images = images_of(&turn.input[s.absorbed.len()..]);
            match s.send_prompt(&delta, &images, &turn.names) {
                Ok(id) => {
                    return match s.await_prompt(id, deadline, cancel) {
                        Ok(r) => {
                            s.absorb(turn, &r);
                            trace_turn(
                                "reuse",
                                &s.session_id,
                                delta.chars().count(),
                                Some(&r),
                                None,
                            );
                            session::give_back(s);
                            Ok(r)
                        }
                        Err(e) => {
                            trace_turn(
                                "reuse",
                                &s.session_id,
                                delta.chars().count(),
                                None,
                                Some(&e),
                            );
                            if s.prompt_in_flight {
                                // The prompt may still be live upstream
                                // (dead wire, unsettled cancel): the
                                // session can't be trusted.
                                s.close();
                            } else {
                                // The failed prompt settled upstream —
                                // the session's absorb state is exactly
                                // what matched this turn, so the host's
                                // identical retry re-prompts the same
                                // delta on the warm session instead of
                                // re-billing the whole transcript.
                                session::give_back(s);
                            }
                            Err(e)
                        }
                    };
                }
                Err(_) => {
                    // The prompt never reached the child (dead pipe): close
                    // it and replay the whole transcript on a fresh session
                    // — the request is still answerable.
                    s.close();
                    reason = "send_failed";
                    spawn_fresh(reason)?
                }
            }
        }
        None => spawn_fresh(reason)?,
    };
    let mode = format!("fresh:{reason}");
    let id = match s.send_prompt(&prompt_text, &images_of(&turn.input), &turn.names) {
        Ok(id) => id,
        Err(e) => {
            // The prompt never reached the child (dead pipe): nothing to
            // pool, and no fresh fallback left — the turn fails.
            trace_turn(&mode, &s.session_id, chars, None, Some(&e));
            s.close();
            return Err(e);
        }
    };
    match s.await_prompt(id, deadline, cancel) {
        Ok(r) => {
            s.absorb(turn, &r);
            trace_turn(&mode, &s.session_id, chars, Some(&r), None);
            session::give_back(s);
            Ok(r)
        }
        Err(e) => {
            trace_turn(&mode, &s.session_id, chars, None, Some(&e));
            if s.prompt_in_flight {
                s.close();
            } else {
                // The failed prompt settled upstream: record the input as
                // absorbed with no reply on record — an identical retry
                // still respawns (nothing extends it), but a later turn
                // of this conversation continues the warm session with
                // only its new tail, and no other conversation can claim
                // a child already holding this transcript.
                s.absorb_failed(turn);
                session::give_back(s);
            }
            Err(e)
        }
    }
}

/// One line per turn when `DEVIN_SUB_DEBUG` is set: mode, session, prompt
/// size and usage — never prompt content. Appends (mode 0600) to
/// `<tempdir>/devin-sub-<pid>.log`. `pub(crate)` so the session pool's
/// keepalive can trace under the same convention.
pub(crate) fn trace_turn(
    mode: &str,
    session_id: &str,
    prompt_chars: usize,
    r: Option<&TurnResult>,
    err: Option<&str>,
) {
    if std::env::var_os("DEVIN_SUB_DEBUG").is_none() {
        return;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_default();
    let (usage, stop) = match r {
        Some(r) => (
            format!(
                "{}/{}/{}",
                r.usage.input_tokens, r.usage.cached_tokens, r.usage.output_tokens
            ),
            r.stop.clone(),
        ),
        None => (
            "0/0/0".to_string(),
            match err {
                Some(e) => format!("error:{}", e.chars().take(120).collect::<String>()),
                None => "error".to_string(),
            },
        ),
    };
    let line = format!(
        "{ts}\t{mode}\tsession={session_id}\tprompt_chars={prompt_chars}\tusage={usage}\tstop={stop}\n"
    );
    let path = std::env::temp_dir().join(format!("devin-sub-{}.log", std::process::id()));
    let mut opts = std::fs::OpenOptions::new();
    opts.append(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    if let Ok(mut f) = opts.open(path) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Fold a completed turn into a Responses SSE stream.
pub fn fold_result(r: &TurnResult) -> Result<Vec<u8>, String> {
    let mut sse = Vec::new();
    let emit = |sse: &mut Vec<u8>, payload: &Value| {
        sse.extend_from_slice(b"data: ");
        sse.extend_from_slice(payload.to_string().as_bytes());
        sse.extend_from_slice(b"\n\n");
    };
    let resp_id = format!("resp_{}", rand_hex(12));
    emit(
        &mut sse,
        &json!({"type": "response.created", "response": {"id": resp_id, "model": "", "status": "in_progress"}}),
    );
    let mut item_id = 0;
    if !r.thought.is_empty() {
        item_id += 1;
        let item = json!({"type": "reasoning", "id": format!("rs_{item_id}"),
            "summary": [{"type": "summary_text", "text": r.thought}]});
        emit(
            &mut sse,
            &json!({"type": "response.output_item.added",
                "output_index": item_id - 1, "item": item}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_item.done",
                "output_index": item_id - 1, "item": item}),
        );
    }
    if !r.text.is_empty() {
        item_id += 1;
        let idx = item_id - 1;
        emit(
            &mut sse,
            &json!({"type": "response.output_text.delta", "output_index": idx, "delta": r.text}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_text.done", "output_index": idx, "text": r.text}),
        );
    }
    for (id, name, args) in &r.calls {
        item_id += 1;
        emit(
            &mut sse,
            &json!({"type": "response.output_item.added",
                "output_index": item_id - 1,
                "item": {"type": "function_call", "id": format!("fc_{item_id}"),
                    "call_id": id, "name": name, "arguments": args}}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.function_call_arguments.done",
                "output_index": item_id - 1,
                "item_id": format!("fc_{item_id}"), "call_id": id,
                "name": name, "arguments": args}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_item.done",
                "output_index": item_id - 1,
                "item": {"type": "function_call", "id": format!("fc_{item_id}"),
                    "call_id": id, "name": name, "arguments": args}}),
        );
    }
    let usage_val = json!({"input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
        "total_tokens": r.usage.input_tokens + r.usage.output_tokens,
        "input_tokens_details": {"cached_tokens": r.usage.cached_tokens,
            "cache_creation_tokens": r.usage.cache_write_tokens}});
    let (status, incomplete) = match r.stop.as_str() {
        "incomplete:max_output_tokens" => ("incomplete", json!({"reason": "max_output_tokens"})),
        s => (s, Value::Null),
    };
    emit(
        &mut sse,
        &json!({"type": "response.completed",
            "response": {"id": resp_id, "status": status,
                "incomplete_details": incomplete, "usage": usage_val}}),
    );
    sse.extend_from_slice(b"data: [DONE]\n\n");
    Ok(sse)
}

pub(crate) fn rand_hex(n: usize) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let mut s = format!("{t:08x}{:08x}", std::process::id());
    while s.len() < n {
        s.push('0');
    }
    s[..n].to_string()
}

#[path = "chat_tests.rs"]
#[cfg(test)]
mod tests;
