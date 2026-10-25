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
//!   `tool_call` notifications: execute/read/web calls redirect onto `bash`
//!   and the web tools, and the rest land on a self-explaining bash echo —
//!   no "does not exist" error card ever renders (see
//!   [`redirect_call`], [`blocked_call`]). A project-level `.devin/config.json`
//!   deny list makes every native tool fail closed regardless.
//! * `session/request_permission` is always answered `cancelled`: nothing a
//!   harness turn does may wait on a prompt.
//! * The relay speaks the OpenAI Responses SSE wire the host already streams,
//!   so no host changes are needed: the declared transport points at the
//!   per-turn relay URL and the host POSTs its standard body with the
//!   per-turn bearer.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use base64::Engine as _;
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
    /// The host session this turn belongs to (the `session-id` header):
    /// stamped into persisted pool records so a respawned sidecar of the
    /// same conversation can claim them across a process boundary. Empty
    /// when the host sent none.
    pub gray_session: String,
    /// The serving account — set by the relay from
    /// [`crate::accounts::select`], `None` on the single-account path.
    /// Pool matching keys on it: a session only continues a turn of the
    /// same account.
    pub account: Option<crate::accounts::Account>,
}

impl PreparedTurn {
    /// The account label for pool matching; `""` on the legacy path.
    pub(crate) fn account_label(&self) -> &str {
        self.account
            .as_ref()
            .map(|a| a.label.as_str())
            .unwrap_or("")
    }
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

/// ASCII identifier bound the host's schema constrains tool names to —
/// `settings`' allowlist honors the same rule so junk never names a tool.
pub(crate) fn valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 50
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn check_tool_name(name: &str, seen: &std::collections::HashSet<String>) -> Result<(), String> {
    if !valid_tool_name(name) || seen.contains(name) {
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

/// Render one Responses `input` item to transcript text. Tool calls render
/// as gray_calls blocks and results as explicit records so the model can
/// reference them; reasoning carriers restore nothing on a fresh session
/// but their text still carries the prior answer.
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
        "function_call" => Some(format!("[Assistant]\n{}", calls_block(&[item]))),
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

fn call_object(item: &Value) -> String {
    let name = item.get("name").and_then(Value::as_str).unwrap_or("?");
    let args = match item.get("arguments") {
        Some(Value::String(raw)) => {
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.clone()))
        }
        Some(v) => v.clone(),
        None => json!({}),
    };
    format!(
        "{{\"name\":{},\"arguments\":{}}}",
        Value::String(name.to_string()),
        args
    )
}

fn calls_block(calls: &[&Value]) -> String {
    let body = calls
        .iter()
        .map(|c| call_object(c))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("```gray_calls\n[{body}]\n```")
}

/// Render a Responses `input` array to transcript text.
fn render_items(items: &[Value]) -> String {
    let mut records: Vec<String> = Vec::new();
    let mut open_assistant = false;
    let mut i = 0;
    while i < items.len() {
        let item = &items[i];
        if item_kind(item) == "function_call" {
            let mut calls = vec![item];
            while i + 1 < items.len() && item_kind(&items[i + 1]) == "function_call" {
                i += 1;
                calls.push(&items[i]);
            }
            let block = calls_block(&calls);
            match records.last_mut() {
                Some(last) if open_assistant => {
                    last.push_str("\n\n");
                    last.push_str(&block);
                }
                _ => records.push(format!("[Assistant]\n{block}")),
            }
            open_assistant = false;
        } else if let Some(r) = render_item(item) {
            open_assistant = item_kind(item) == "message"
                && item.get("role").and_then(Value::as_str) == Some("assistant");
            records.push(r);
        }
        i += 1;
    }
    records.join("\n\n")
}

/// Translate an OpenAI Responses body into one funnel turn.
pub fn prepare_turn(
    body: &Value,
    model: &str,
    account_home: Option<&std::path::Path>,
) -> Result<PreparedTurn, String> {
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
    // Responses `reasoning.effort` is the host's /thinking pick; the model
    // id family + tier name resolve to the real native id (the catalog
    // collapses `swe-2-high|medium|max` into one `swe-2` row, so the effort
    // is how a tier actually gets chosen).
    let effort = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str);
    let native_model = crate::catalog::native_model_in(model, effort, account_home);
    system_parts.push(funnel_contract(
        &tool_specs,
        native_model.starts_with("fusion-"),
    ));
    Ok(PreparedTurn {
        system: system_parts.join("\n\n"),
        content_line,
        input,
        names,
        native_model,
        gray_session: String::new(),
        account: None,
    })
}

/// The text funnel contract the model answers with. Verified with Claude
/// Sonnet: it emits the block exactly as specified.
fn funnel_contract(tool_specs: &[String], fusion: bool) -> String {
    let mut s = String::from("You are the model behind another agent harness (\"gray\"). ");
    s.push_str(if fusion {
        "Your own built-in file/process/web tools are DISABLED in this session: such calls auto-redirect onto \
        `bash` and the host's web tools, anything else returns an explanation instead of running. \
        Never call them yourself. The one exception is your sidekick: your built-in handoff to your \
        paired sidekick model is ENABLED, and it is the only thing \"sidekick\" means here."
    } else {
        "Your own built-in tools are \
        DISABLED in this session: file/process/web calls auto-redirect onto `bash` and the host's \
        web tools, anything else returns an explanation instead of running — a native call never \
        does what you asked. Never call them."
    });
    s.push_str("\n\
        Instead, the harness executes tools for you. To call harness tools, end your reply with \
        exactly one fenced block holding every call for this round, for example:\n\n```gray_calls\n\
        [{\"name\": \"bash\", \"arguments\": {\"command\": \"rg -n parse_config src\"}},\n \
        {\"name\": \"bash\", \"arguments\": {\"command\": \"sed -n 1,80p src/config.rs\"}},\n \
        {\"name\": \"bash\", \"arguments\": {\"command\": \"git log --oneline -5 -- src/config.rs\"}}]\n```\n\n\
        Batch by default. Before writing the block, list everything you can already look up \
        without waiting for a result (files to read, searches, logs, status checks) and give \
        each its own call in this one block: they run concurrently and cost one round together. \
        Defer a call only when it needs an earlier call's output. Prefer several separate calls \
        over one long `&&`/`;`-chained command; chain only steps that depend on each other \
        (`cd dir && make`). \
        Do not write anything after the block. \
        If no tool is needed, answer normally with no block. \
        Never call a tool just to wait, poll or pass a round (`true`, `:`, `sleep`, `echo`): \
        a command that outlives its timeout keeps running as a background job — end your reply \
        to be woken when it finishes, `tail` its log to peek, `kill -- -<pgid>` to stop it; \
        otherwise reply with no block. \
        Later messages in this conversation come from the harness: tool results \
        arrive as `[Tool result id=…]` blocks and new user input as `[User]` blocks.\n\
        You write exactly ONE assistant turn. Never write `[User]`, `[Tool result …]` or any other \
        transcript record yourself; tool results only come from the harness. Never claim a command \
        ran unless its `[Tool result]` is in the transcript.\n");
    s.push_str(if fusion {
        "When the user asks for the sidekick, or a task splits off an independent piece, hand it to \
        your sidekick with your built-in sidekick handoff and wait for its report. Never use \
        `gray subagents`, the `subagent*` harness tools, or any other background agent for sidekick \
        work: those are separate agents, not your sidekick. The sidekick's commands and file \
        reads/writes run inline in the user's workspace through the harness, so its results are \
        real. Make your own tool calls with the gray_calls block."
    } else {
        "To delegate parallel or independent work, spawn a gray subagent through `bash`: \
        `gray subagents run --name NAME 'task'` (add `--isolation worktree` when parallel \
        runs edit the same repo); follow up with `gray subagents status`, `gray subagents \
        view ID [--lines N]`, `gray subagents steer ID 'follow-up'`, `gray subagents stop ID`. \
        Your own native subagent tool is disabled — a child spawned inside this session can \
        never run a tool, so delegate via `gray subagents` only."
    });
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
/// tail, so a prefix parse recovers it and the markup is dropped. The
/// array is parsed from the whole tail, never cut at the first closing
/// fence: a command may carry its own fence (a heredoc writing markdown).
/// A strict parse that fails gets one retry through [`repair_json`].
pub fn parse_calls_block(text: &str, names: &[String]) -> Option<(String, Vec<(String, String)>)> {
    const FENCE: &str = "```gray_calls";
    // Scan occurrences last-to-first; the first real block wins. A real
    // block's info string ends at the newline (or EOF) and its body leads
    // with `[` — an inline mention (`the <fence> contract)`) or a fenced
    // explanation is prose about the funnel, not a call, so it stays in
    // the text where the host's leaked-markup nudge can answer it.
    let mut search_from = text.len();
    let (start, candidate, parsed) = loop {
        let idx = text[..search_from].rfind(FENCE)?;
        let after = &text[idx + FENCE.len()..];
        let info_ends = after.is_empty() || after.starts_with('\n');
        let body = after.trim_start();
        if info_ends && body.starts_with('[') {
            // A JSON array is self-delimiting, so it is parsed before any
            // closing fence is looked for: a fence inside one of its strings
            // (a command grepping for the funnel's own marker) must not cut
            // the block short. Only an array that fails to parse falls back
            // to the fence-delimited text, for the guess path below.
            let fenced = after
                .find("```")
                .map_or(after, |close| &after[..close])
                .trim();
            let parsed = json_array_prefix(body)
                .or_else(|| json_array_prefix(&repair_json(body)))
                .or_else(|| {
                    [body, fenced].into_iter().find_map(|s| {
                        close_brackets(&repair_json(s)).and_then(|s| json_array_prefix(&s))
                    })
                });
            let body = if parsed.is_some() { body } else { fenced };
            break (idx, body, parsed);
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
    let calls: Vec<(String, String)> = match parsed {
        Some(arr) => arr
            .iter()
            .map(|c| {
                let name = c.get("name").and_then(Value::as_str).unwrap_or(guess);
                let args = c.get("arguments").unwrap_or(&Value::Null);
                // The advertised set is the whole permit: a name outside
                // `names` can't run even when the host owns such a tool —
                // fold to `native__…` so the "does not exist" error
                // teaches the real surface instead of silently executing.
                (funnel_name(name, names), args.to_string())
            })
            .collect(),
        None => vec![(guess.to_string(), candidate.to_string())],
    };
    Some((text[..start].trim_end().to_string(), calls))
}

pub(crate) fn funnel_name(name: &str, names: &[String]) -> String {
    if names.iter().any(|n| n == name) {
        name.to_string()
    } else {
        denamespace(name, names).unwrap_or_else(|| native_name(name))
    }
}

fn json_value_prefix(s: &str) -> Option<(Value, usize)> {
    let mut stream = serde_json::Deserializer::from_str(s).into_iter::<Value>();
    let v = stream.next()?.ok()?;
    Some((v, stream.byte_offset()))
}

/// A leading self-delimiting JSON array, ignoring whatever follows.
fn json_array_prefix(s: &str) -> Option<Vec<Value>> {
    match json_value_prefix(s)? {
        (Value::Array(a), _) => Some(a),
        _ => None,
    }
}

fn tool_call_header(s: &str) -> Option<(&str, usize)> {
    let mut off = 0;
    for line in s.split_inclusive('\n') {
        let head = line.trim_start_matches([' ', '\t']);
        if let Some(inner) = head.strip_prefix("[Tool call ") {
            return Some((inner.trim_end(), off + line.len()));
        }
        off += line.len();
    }
    None
}

pub(crate) fn salvage_calls(text: &str, names: &[String]) -> Vec<(String, String)> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some((header, end)) = tool_call_header(rest) {
        let name = header
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches(']');
        rest = &rest[end..];
        if name.is_empty() {
            continue;
        }
        let mut body = rest.trim_start();
        if let Some(fenced) = body.strip_prefix("```") {
            body = fenced.strip_prefix("json").unwrap_or(fenced).trim_start();
        }
        match json_value_prefix(body) {
            Some((args, used)) => {
                calls.push((funnel_name(name, names), args.to_string()));
                rest = &body[used..];
            }
            None => calls.push((funnel_name(name, names), "{}".to_string())),
        }
    }
    calls
}

/// Insert the closers a model dropped: a `]` or `}` that doesn't match the
/// innermost open bracket first closes what is still open inside it, and an
/// array left open at the end gets its closers appended. `None` when nothing
/// needed fixing, a closer has nothing to close, or the text ends inside a
/// string — the slip is then not a missing bracket.
fn close_brackets(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len() + 8);
    let mut stack: Vec<char> = Vec::new();
    let (mut in_str, mut escaped, mut fixed) = (false, false, false);
    for (i, c) in s.char_indices() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            ']' | '}' => {
                while stack.last().is_some_and(|&want| want != c) {
                    out.extend(stack.pop());
                    fixed = true;
                }
                stack.pop()?;
                if stack.is_empty() {
                    out.push(c);
                    out.push_str(&s[i + c.len_utf8()..]);
                    return fixed.then_some(out);
                }
            }
            _ => {}
        }
        out.push(c);
    }
    if in_str || stack.is_empty() {
        return None;
    }
    out.extend(stack.into_iter().rev());
    Some(out)
}

/// Undo the slips models make inside JSON strings: raw control characters
/// (a heredoc's literal newlines and tabs) get escaped, and escapes JSON
/// doesn't define (a shell dollar, a regex digit class) keep their
/// backslash as a literal one. Structure is never touched.
fn repair_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut chars = s.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if !in_str {
            in_str = c == '"';
            out.push(c);
            continue;
        }
        match c {
            '"' => {
                in_str = false;
                out.push(c);
            }
            '\\' => match chars.peek() {
                Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') => {
                    out.push(c);
                    out.extend(chars.next());
                }
                _ => out.push_str("\\\\"),
            },
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out
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
        // carries `rawInput.url` and `"webfetch"`), and reports its local
        // `grep` as `kind: "search"` with the same {query}-shaped payload —
        // so `kind` is a hint, never a verdict. A frame that names a tool
        // resolves by that name (web names included); the payload guess is
        // for anonymous frames only.
        "search" | "fetch" => match native_tool_name(update).or_else(|| title_tool(update)) {
            Some(_) => name_call(update, input, names).or_else(|| title_call(update, input, names)),
            None => web_redirect(update, input, names),
        },
        // kind:"other" is the grab-bag: MCP-bridge and native-tool calls
        // whose title ("Calling bash from harness") still names the tool
        // the model meant even when rawInput doesn't carry tool_name.
        // When the title says nothing either, the payload itself still
        // does — "Wrote /x" with {file_path, content} is a write.
        _ => name_call(update, input, names)
            .or_else(|| title_call(update, input, names))
            .or_else(|| shape_call(update, input, names)),
    }
}

/// Title and kind said nothing, but the payload still carries intent:
/// {command} is an execute, {file_path} a read — or a write/edit when
/// content rides along — {url|query} a web call. Read_shell-family
/// builtins and "Wrote X" updates used to die as native__ noise even
/// though their rawInput is exactly what a named kind would carry.
fn shape_call(update: &Value, input: &Value, names: &[String]) -> Option<(String, String)> {
    if !input.is_object() {
        return None;
    }
    let has = |n: &str| names.iter().any(|x| x == n);
    if let Some(cmd) = input.get("command").and_then(Value::as_str) {
        return has("bash").then(|| ("bash".to_string(), json!({"command": cmd}).to_string()));
    }
    if input.get("file_path").is_some() {
        let args = if input.get("old_string").is_some() {
            edit_redirect(input)
        } else if input.get("content").is_some() || input.get("new_string").is_some() {
            write_redirect(&json!({
                "file_path": input["file_path"],
                "content": input.get("content").or_else(|| input.get("new_string")).cloned().unwrap_or(Value::Null),
            }))
        } else {
            input
                .get("file_path")
                .and_then(Value::as_str)
                .map(|p| json!({"command": format!("cat -- {}", shell_quote(p))}).to_string())
        };
        return args.and_then(|a| bash_args(a, has("bash")));
    }
    if input.get("url").is_some() || input.get("query").is_some() {
        // Shape guessing is for anonymous frames: a named call that reached
        // this point failed to resolve — folding it into web_search on
        // payload alone turns e.g. devin's `grep` ({query, path} — a
        // ripgrep pattern, not a web query) into a Tavily search.
        if native_tool_name(update).is_some() || title_tool(update).is_some() {
            return None;
        }
        return web_redirect(update, input, names);
    }
    None
}

/// The native tool a `tool_call` update was for, from the frame itself:
/// `_meta.inferenceToolName`, else the `name:` prefix of toolCallId
/// ("web_search:1#db12" -> web_search). Always the stable tool name —
/// never the display title, so paths and queries can't leak into
/// blocked-call names (`request_scope` titles carry the target path).
fn native_tool_name(update: &Value) -> Option<String> {
    if let Some(t) = update
        .pointer("/_meta/cognition.ai~1inferenceToolName")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    {
        return Some(t.to_string());
    }
    update
        .get("toolCallId")
        .and_then(Value::as_str)
        .and_then(|id| id.split_once(':').map(|(t, _)| t))
        .filter(|t| {
            t.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        })
        .map(String::from)
}

/// Resolve a reached-for tool name: web tools route through web_redirect,
/// advertised names execute, namespaced names strip down.
fn resolve_tool(
    tool: &str,
    update: &Value,
    input: &Value,
    names: &[String],
) -> Option<(String, String)> {
    if matches!(tool, "web_search" | "web_fetch" | "webfetch") {
        return web_redirect(update, input, names);
    }
    if names.iter().any(|n| n == tool) {
        return Some((tool.to_string(), input.to_string()));
    }
    denamespace(tool, names).map(|t| (t, input.to_string()))
}

/// The tool a native update was reaching for, parsed out of its title.
/// "Calling ctx_exec from apify" -> ctx_exec; "mcp__srv__tool" -> tool;
/// anything else -> None.
fn title_tool(update: &Value) -> Option<String> {
    let title = update.get("title").and_then(Value::as_str)?;
    if let Some(rest) = title.strip_prefix("Calling ") {
        let tool = rest.split(" from ").next()?;
        if !tool.is_empty() && tool.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Some(tool.to_string());
        }
        return None;
    }
    title
        .strip_prefix("mcp__")
        .and_then(|t| t.rsplit("__").next())
        .filter(|t| !t.is_empty())
        .map(String::from)
        .or_else(|| {
            // A bare "tool_name" title — some harnesses label the update
            // with nothing but the tool's name.
            (!title.is_empty()
                && title.len() <= 50
                && title
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .then(|| title.to_string())
        })
}

/// A title-named call whose tool is on the advertised surface becomes the
/// call the model meant. web_* goes through web_redirect so a disallowed
/// fetch still lands on curl; anything else gets rawInput passed as its
/// args — a shape mismatch surfaces as the tool's own error, which still
/// teaches more than a dead-end rejection.
fn title_call(update: &Value, input: &Value, names: &[String]) -> Option<(String, String)> {
    resolve_tool(&title_tool(update)?, update, input, names)
}

/// Frame-named call: the update carries the stable tool name outright.
fn name_call(update: &Value, input: &Value, names: &[String]) -> Option<(String, String)> {
    resolve_tool(&native_tool_name(update)?, update, input, names)
}

/// "plugin_ctx_exec" / "harness_bash": the model namespaces an advertised
/// tool. Strip the prefix and retry the advertised check before folding
/// to native__.
fn denamespace(name: &str, names: &[String]) -> Option<String> {
    for pre in ["plugin_", "harness_", "gray_"] {
        if let Some(rest) = name.strip_prefix(pre)
            && names.iter().any(|n| n == rest)
        {
            return Some(rest.to_string());
        }
    }
    None
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
/// it: when `bash` is on the surface it lands on an echo that explains
/// itself — "X is a disabled built-in tool — nothing ran" — so the turn
/// stays alive, the transcript says why, and no error card ever renders.
/// Bash-less surfaces keep the `native__` fold: the host's "does not
/// exist" names the true surface instead of dropping the turn.
pub(crate) fn blocked_call(update: &Value, names: &[String]) -> (String, String) {
    let raw = native_tool_name(update)
        .or_else(|| title_tool(update))
        .or_else(|| update.get("kind").and_then(Value::as_str).map(String::from))
        .or_else(|| {
            update
                .get("title")
                .and_then(Value::as_str)
                .map(String::from)
        })
        .unwrap_or_else(|| "tool_call".to_string());
    if names.iter().any(|n| n == "bash") {
        let msg = format!(
            "{raw} is a disabled built-in tool in this harness — nothing ran. \
             Do file/process work through bash, or use an advertised tool."
        );
        return (
            "bash".to_string(),
            json!({"command": format!("printf '%s\\n' {}", shell_quote(&msg))}).to_string(),
        );
    }
    (native_name(&raw), "{}".to_string())
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

fn working_directory(system: &str) -> Option<String> {
    system.lines().find_map(|line| {
        line.strip_prefix("Working directory: ")
            .and_then(|value| serde_json::from_str::<String>(value).ok())
    })
}

fn mapped_stage_path(value: &str, stage_work: &Path, system: &str) -> String {
    let Some(working_directory) = working_directory(system) else {
        return value.to_string();
    };
    let stage_work = stage_work.to_string_lossy();
    value.replace(stage_work.as_ref(), &working_directory)
}

pub(crate) fn sidekick_terminal_command(
    command: &str,
    args: &[String],
    cwd: Option<&str>,
    stage_work: &Path,
    system: &str,
) -> String {
    let command = mapped_stage_path(command, stage_work, system);
    let invocation = if args.is_empty() {
        command
    } else {
        let args = args
            .iter()
            .map(|arg| shell_quote(&mapped_stage_path(arg, stage_work, system)))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {args}", shell_quote(&command))
    };
    let mapped_cwd = cwd
        .filter(|cwd| !cwd.is_empty())
        .map(|cwd| mapped_stage_path(cwd, stage_work, system));
    let already_at_working_directory = working_directory(system)
        .as_deref()
        .is_some_and(|work| mapped_cwd.as_deref() == Some(work));
    match mapped_cwd.filter(|_| !already_at_working_directory) {
        Some(cwd) => format!("cd -- {} && {invocation}", shell_quote(&cwd)),
        None => invocation,
    }
}

pub(crate) fn sidekick_read_command(
    path: &str,
    line: Option<u64>,
    limit: Option<u64>,
    stage_work: &Path,
    system: &str,
) -> String {
    let path = mapped_stage_path(path, stage_work, system);
    match (line, limit) {
        (None, None) => format!("cat -- {}", shell_quote(&path)),
        (line, limit) => {
            let start = line.unwrap_or(1).max(1);
            let end = limit.map(|limit| start.saturating_add(limit.max(1) - 1));
            let range = end.map_or_else(|| format!("{start},$p"), |end| format!("{start},{end}p"));
            format!("sed -n {} -- {}", shell_quote(&range), shell_quote(&path))
        }
    }
}

pub(crate) fn sidekick_write_command(
    path: &str,
    content: &str,
    stage_work: &Path,
    system: &str,
) -> String {
    let path = mapped_stage_path(path, stage_work, system);
    let encoded = base64::engine::general_purpose::STANDARD.encode(content.as_bytes());
    format!(
        "mkdir -p -- \"$(dirname -- {})\" && printf %s {} | base64 -d > {}",
        shell_quote(&path),
        shell_quote(&encoded),
        shell_quote(&path)
    )
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

pub(crate) fn batch_cut_note(dropped: usize) -> String {
    format!(
        "[Harness note] Your previous step issued {} tool calls at once; one hit a native tool that \
        is rejected upstream, which cancelled the other {dropped} before the harness saw them, so \
        they did NOT run. Re-issue the ones you still need in one gray_calls block.",
        dropped + 1
    )
}

pub(crate) const CUT_NOTE: &str = "[Harness note] Your previous reply was cut where you started writing a transcript record yourself; everything from that point on was discarded, never delivered, and none of it ran. Write exactly one assistant turn.";

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
    continuation_parts(absorbed, reply_call_ids, reply_text, input).map(|(delta, _)| delta)
}

pub(crate) fn continuation_output(
    absorbed: &[Value],
    reply_call_ids: &[String],
    reply_text: &str,
    input: &[Value],
    expected_call_id: &str,
) -> Result<(String, String), &'static str> {
    let (delta, tail) = continuation_parts(absorbed, reply_call_ids, reply_text, input)?;
    if tail.iter().any(|item| {
        item_kind(item) != "function_call_output"
            || item.get("call_id").and_then(Value::as_str) != Some(expected_call_id)
    }) {
        return Err("parked_tail");
    }
    let output = tail
        .iter()
        .map(|item| match item.get("output") {
            Some(Value::String(text)) => text.clone(),
            Some(value) => value.to_string(),
            None => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok((delta, output))
}

fn continuation_parts<'a>(
    absorbed: &[Value],
    reply_call_ids: &[String],
    reply_text: &str,
    input: &'a [Value],
) -> Result<(String, &'a [Value]), &'static str> {
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
        return Ok((format!("{UNDELIVERED_NOTE}\n\n{delta}"), tail));
    }
    Ok((delta, tail))
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
        reasoning_tokens: get("thoughtTokens"),
    };
    if usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_tokens == 0
        && usage.cache_write_tokens == 0
        && usage.reasoning_tokens == 0
    {
        None
    } else {
        Some(usage)
    }
}

/// What one `devin acp` session produced.
#[derive(Clone)]
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
    /// `thoughtTokens` from the ACP end-turn usage, when the model reports it.
    pub reasoning_tokens: usize,
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
    let r = run_turn_inner(turn, cancel, timeout);
    // Account-scoped failures bench the serving account (a success
    // clears a stale bench); request-scoped ones leave it alone.
    crate::accounts::observe(turn.account.as_ref(), &r);
    r
}

fn run_turn_inner(
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
    if let Some(r) = session::replay(turn) {
        trace_turn("replay", "-", chars, Some(&r), None, None);
        return Ok(r);
    }
    let (mut claimed, mut reason) = session::take(turn, cancel);
    // Nothing live continues this turn: an orphaned session of a dead
    // sidecar (a gray restart) may, loaded instead of re-billed.
    let mut mode = "reuse";
    if claimed.is_none() && !matches!(reason, "keepalive_busy" | "parked_mismatch") {
        claimed = session::reattach(turn, deadline, cancel);
        if claimed.is_some() {
            mode = "reload";
        }
    }
    let spawn_fresh = |reason: &str| match session::spawn(turn, deadline, cancel) {
        Ok(s) => Ok(s),
        Err(e) => {
            trace_turn(&format!("fresh:{reason}"), "-", chars, None, Some(&e), None);
            Err(e)
        }
    };
    let mut s = match claimed {
        Some((mut s, delta)) => {
            if s.is_parked() {
                let parked_mode = format!("{mode}:sidekick_resume");
                match s.resume_parked(turn, deadline, cancel) {
                    Ok(session::AwaitPrompt::Complete(r)) => {
                        s.absorb(turn, &r);
                        trace_turn(
                            &parked_mode,
                            &s.session_id,
                            delta.chars().count(),
                            Some(&r),
                            None,
                            None,
                        );
                        session::give_back(s);
                        return Ok(r);
                    }
                    Ok(session::AwaitPrompt::Parked(r)) => {
                        s.absorb(turn, &r);
                        trace_turn(
                            &format!("{mode}:sidekick_park"),
                            &s.session_id,
                            delta.chars().count(),
                            Some(&r),
                            None,
                            None,
                        );
                        session::give_back(s);
                        return Ok(r);
                    }
                    Err(e) => {
                        trace_turn(
                            &parked_mode,
                            &s.session_id,
                            delta.chars().count(),
                            None,
                            Some(&e),
                            None,
                        );
                        if s.prompt_in_flight {
                            s.close();
                        } else {
                            session::give_back(s);
                        }
                        return Err(e);
                    }
                }
            }
            let delta = s.note_cut(delta);
            let images = images_of(&turn.input[s.absorbed.len()..]);
            match s.send_prompt(&delta, &images, &turn.names) {
                Ok(id) => match s.await_prompt_or_parked(id, deadline, cancel) {
                    Ok(session::AwaitPrompt::Complete(r)) => {
                        s.absorb(turn, &r);
                        trace_turn(
                            mode,
                            &s.session_id,
                            delta.chars().count(),
                            Some(&r),
                            None,
                            None,
                        );
                        session::give_back(s);
                        return Ok(r);
                    }
                    Ok(session::AwaitPrompt::Parked(r)) => {
                        s.absorb(turn, &r);
                        trace_turn(
                            &format!("{mode}:sidekick_park"),
                            &s.session_id,
                            delta.chars().count(),
                            Some(&r),
                            None,
                            None,
                        );
                        session::give_back(s);
                        return Ok(r);
                    }
                    Err(e) => {
                        trace_turn(
                            mode,
                            &s.session_id,
                            delta.chars().count(),
                            None,
                            Some(&e),
                            None,
                        );
                        if s.rewarmed {
                            // The re-pooled child's first prompt never
                            // settled — a stale record, not a dead
                            // conversation: the request is still
                            // answerable on a fresh child.
                            s.close();
                            reason = "rewarm_dead";
                            spawn_fresh(reason)?
                        } else if s.prompt_in_flight {
                            // The prompt may still be live upstream
                            // (dead wire, unsettled cancel): the
                            // session can't be trusted.
                            s.close();
                            return Err(e);
                        } else {
                            // The failed prompt settled upstream —
                            // the session's absorb state is exactly
                            // what matched this turn, so the host's
                            // identical retry re-prompts the same
                            // delta on the warm session instead of
                            // re-billing the whole transcript.
                            session::give_back(s);
                            return Err(e);
                        }
                    }
                },
                Err(_) => {
                    // The prompt never reached the child (dead pipe): close
                    // it and replay the whole transcript on a fresh session
                    // — the request is still answerable.
                    reason = if s.rewarmed {
                        "rewarm_dead"
                    } else {
                        "send_failed"
                    };
                    s.close();
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
            trace_turn(&mode, &s.session_id, chars, None, Some(&e), None);
            s.close();
            return Err(e);
        }
    };
    match s.await_prompt_or_parked(id, deadline, cancel) {
        Ok(session::AwaitPrompt::Complete(r)) => {
            s.absorb(turn, &r);
            trace_turn(&mode, &s.session_id, chars, Some(&r), None, None);
            session::give_back(s);
            Ok(r)
        }
        Ok(session::AwaitPrompt::Parked(r)) => {
            s.absorb(turn, &r);
            trace_turn(
                "fresh:sidekick_park",
                &s.session_id,
                chars,
                Some(&r),
                None,
                None,
            );
            session::give_back(s);
            Ok(r)
        }
        Err(e) => {
            trace_turn(&mode, &s.session_id, chars, None, Some(&e), None);
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
    diag: Option<&str>,
) {
    // Fresh spawns and failures always land in the persistent pool log:
    // each one is a whole-transcript re-bill, and without the reason a
    // cache miss can't be told apart from an eviction or a diverged prefix.
    // Reloads land too — each is a re-bill avoided, the number to watch.
    // Reuse turns (the normal case) stay behind DEVIN_SUB_DEBUG.
    let cold = mode == "reuse"
        && r.is_some_and(|r| {
            r.usage.input_tokens >= 20_000 && r.usage.cached_tokens * 2 < r.usage.input_tokens
        });
    let persist =
        cold || mode.starts_with("fresh") || mode == "reload" || mode == "replay" || err.is_some();
    let debug = std::env::var_os("DEVIN_SUB_DEBUG").is_some();
    if !persist && !debug {
        return;
    }
    let mode = if cold { "reuse:cold" } else { mode };
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
    let mut line = format!(
        "{ts}\t{mode}\tsession={session_id}\tprompt_chars={prompt_chars}\tusage={usage}\tstop={stop}"
    );
    if let Some(d) = diag {
        line.push_str(&format!("\tdiag={d}"));
    }
    line.push('\n');
    if persist {
        append_pool_log(&line);
    }
    if debug {
        let path = std::env::temp_dir().join(format!("devin-sub-{}.log", std::process::id()));
        append_private(&path, &line);
    }
}

/// One pool lifecycle line (reap/evict) in the persistent pool log: the
/// reason a later turn finds `no_session` and spawns fresh.
pub(crate) fn pool_event(event: &str, session_id: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_default();
    append_pool_log(&format!("{ts}\t{event}\tsession={session_id}\n"));
}

/// The persistent pool log rotates past this size (one `.1` kept).
const POOL_LOG_MAX: u64 = 1 << 20;

/// `<gray home>/devin-sub/pool.log`, pid-tagged so concurrent sidecars
/// (one per gray process) stay distinguishable. Best-effort.
fn append_pool_log(line: &str) {
    let Some(dir) = gray_core::paths::gray_home().map(|h| h.join("devin-sub")) else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("pool.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > POOL_LOG_MAX) {
        let _ = std::fs::rename(&path, dir.join("pool.log.1"));
    }
    append_private(&path, &format!("pid={}\t{line}", std::process::id()));
}

/// Append to a 0600 file, creating it.
fn append_private(path: &std::path::Path, line: &str) {
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
    let reasoning_tokens = if r.usage.reasoning_tokens > 0 {
        r.usage.reasoning_tokens
    } else {
        r.thought.len().div_ceil(4)
    };
    let reasoning_tokens = reasoning_tokens.min(r.usage.output_tokens);
    let usage_val = json!({"input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
        "total_tokens": r.usage.input_tokens + r.usage.output_tokens,
        "input_tokens_details": {"cached_tokens": r.usage.cached_tokens,
            "cache_creation_tokens": r.usage.cache_write_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens}});
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

pub fn rand_hex(n: usize) -> String {
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
