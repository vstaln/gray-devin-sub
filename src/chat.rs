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
//!   and the model answers with a fenced ```gray_calls block; models that
//!   ignore the text funnel and reach for a native tool anyway are caught by
//!   `tool_call` notifications and redirected to `bash` (see
//!   [`redirect_call`]). A project-level `.devin/config.json` deny list makes
//!   every native tool fail closed regardless.
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
                Some("input_image" | "image" | "input_file") => Some("[image omitted]".to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
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
    Ok(PreparedTurn {
        system: system_parts.join("\n\n"),
        content_line,
        input,
        names,
        native_model: crate::catalog::native_model(model),
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

/// Parse the funnel: the LAST ```gray_calls fenced block in `text`.
/// Valid = a JSON array of objects with `name` in the host tool set and
/// object `arguments`. Anything malformed/foreign → `None` (the caller
/// treats the whole reply as plain text, never invented calls).
///
/// The closing fence is optional: SWE models often terminate the call
/// list with their native pipe-delimited tool markup instead of ``` — or
/// emit no close at all. In both cases the JSON array still leads the
/// tail, so a prefix parse recovers it and the markup is dropped.
pub fn parse_calls_block(text: &str, names: &[String]) -> Option<(String, Vec<(String, String)>)> {
    const FENCE: &str = "```gray_calls";
    let start = text.rfind(FENCE)?;
    let after = &text[start + FENCE.len()..];
    let candidate = match after.find("```") {
        Some(close) => &after[..close],
        None => after,
    }
    .trim();
    let arr = json_array_prefix(candidate)?;
    let mut calls: Vec<(String, String)> = Vec::new();
    for c in &arr {
        let name = c.get("name").and_then(Value::as_str)?;
        if !names.iter().any(|n| n == name) {
            return None;
        }
        let args = c.get("arguments")?;
        if !args.is_object() {
            return None;
        }
        calls.push((name.to_string(), serde_json::to_string(args).ok()?));
    }
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

/// Redirect a native `tool_call` notification onto `bash`, when the host
/// tools include it. Verified shapes: `{"kind":"execute","rawInput":
/// {"command": "..."}}` and `{"kind":"read","rawInput":{"file_path":"..."}}`.
pub(crate) fn redirect_call(update: &Value, has_bash: bool) -> Option<String> {
    if !has_bash {
        return None;
    }
    let kind = update.get("kind").and_then(Value::as_str).unwrap_or("");
    let input = update.get("rawInput").unwrap_or(&Value::Null);
    match kind {
        "execute" => input
            .get("command")
            .and_then(Value::as_str)
            .map(|c| json!({"command": c}).to_string()),
        "read" => input
            .get("file_path")
            .and_then(Value::as_str)
            .map(|p| json!({"command": format!("cat -- {}", shell_quote(p))}).to_string()),
        _ => None,
    }
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
    if call_ids.as_slice() != reply_call_ids || texts.join("\n").trim() != reply_text.trim() {
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
    };
    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.cached_tokens == 0 {
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
}

#[derive(Default, Clone)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_tokens: usize,
}

/// Run one turn: prompt a pooled continuation session with only the delta
/// when this request's history strictly extends what it last answered,
/// else spawn `devin acp`, drive the ACP handshake and prompt with
/// `system` + the whole transcript. `timeout` is the whole-turn budget;
/// `cancel` is set when the relay client went away mid-turn.
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
    let (claimed, mut reason) = session::take(turn);
    let spawn_fresh = |reason: &str| match session::spawn(turn, deadline, cancel) {
        Ok(s) => Ok(s),
        Err(e) => {
            trace_turn(&format!("fresh:{reason}"), "-", chars, None, Some(&e));
            Err(e)
        }
    };
    let mut s = match claimed {
        Some((mut s, delta)) => {
            match s.send_prompt(&delta, &turn.names) {
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
                            s.close();
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
    match s
        .send_prompt(&prompt_text, &turn.names)
        .and_then(|id| s.await_prompt(id, deadline, cancel))
    {
        Ok(r) => {
            s.absorb(turn, &r);
            trace_turn(&mode, &s.session_id, chars, Some(&r), None);
            session::give_back(s);
            Ok(r)
        }
        Err(e) => {
            trace_turn(&mode, &s.session_id, chars, None, Some(&e));
            s.close();
            Err(e)
        }
    }
}

/// One line per turn when `DEVIN_SUB_DEBUG` is set: mode, session, prompt
/// size and usage — never prompt content. Appends (mode 0600) to
/// `<tempdir>/devin-sub-<pid>.log`.
fn trace_turn(
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
pub fn fold_result(r: &TurnResult, names: &[String]) -> Result<Vec<u8>, String> {
    for (_, name, _) in &r.calls {
        if !names.iter().any(|n| n == name) {
            return Err(format!(
                "native requested a tool outside the current host inventory: {name:?}"
            ));
        }
    }
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
        emit(
            &mut sse,
            &json!({"type": "response.output_text.delta", "output_index": 0, "delta": r.text}),
        );
        emit(
            &mut sse,
            &json!({"type": "response.output_text.done", "output_index": 0, "text": r.text}),
        );
    }
    let usage_val = json!({"input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
        "total_tokens": r.usage.input_tokens + r.usage.output_tokens,
        "input_tokens_details": {"cached_tokens": r.usage.cached_tokens}});
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
