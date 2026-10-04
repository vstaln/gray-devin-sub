//! One-shot chat turn over the loopback relay: the OpenAI Responses body the
//! host POSTs becomes ONE `devin acp` session (initialize → session/new →
//! session/prompt → session/delete) whose native tools are fully denied.
//!
//! The funnel contract (verified against `devin acp` behaviour):
//! * ACP has no system role and no assistant replay: the funnel system text
//!   (contract + tool manifest + host instructions) leads the single prompt
//!   text block, then the labeled transcript. One upstream request per turn.
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

use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::setup;

/// Relay rejection when native retries past the single admitted request.
pub const ADMISSION_CONSUMED: &str = "DEVIN_MODEL_ADMISSION_CONSUMED";

/// Deny list written into both the project config and the user config the
/// child sees: every native capability Devin exposes.
const DENY_RULES: &[&str] = &[
    "read",
    "edit",
    "grep",
    "glob",
    "exec",
    "Fetch(domain:*)",
    "mcp__*",
];

/// One translated turn: funnel system text, transcript, host tool names.
pub struct PreparedTurn {
    pub system: String,
    pub content_line: String,
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
/// reasoning carriers restore nothing (one fresh session per turn) but
/// their text still carries the prior answer.
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
    if !matches!(last_kind, "message" | "function_call" | "function_call_output") {
        return Err(
            "history must end in a nonempty user/tool-result message; assistant prefill is unsupported"
                .into(),
        );
    }
    let mut parts: Vec<String> = Vec::new();
    for item in &input {
        if let Some(text) = render_item(item) {
            parts.push(text);
        }
    }
    if parts.is_empty() {
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
        content_line: parts.join("\n\n"),
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
        If no tool is needed, answer normally with no block.",
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
pub fn parse_calls_block(
    text: &str,
    names: &[String],
) -> Option<(String, Vec<(String, String)>)> {
    const FENCE: &str = "```gray_calls";
    let start = text.rfind(FENCE)?;
    let after = &text[start + FENCE.len()..];
    let close = after.find("```")?;
    let block = after[..close].trim();
    let arr: Vec<Value> = serde_json::from_str(block).ok()?;
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

/// Redirect a native `tool_call` notification onto `bash`, when the host
/// tools include it. Verified shapes: `{"kind":"execute","rawInput":
/// {"command": "..."}}` and `{"kind":"read","rawInput":{"file_path":"..."}}`.
fn redirect_call(update: &Value, has_bash: bool) -> Option<String> {
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

/// Run one turn: spawn `devin acp`, drive the ACP handshake, prompt, fold.
/// `timeout` is the whole-turn budget.
pub fn spawn_turn(turn: &PreparedTurn, timeout: Duration) -> Result<TurnResult, String> {
    let binary = setup::resolve_command().ok_or_else(|| setup::INSTALL_HINT.to_string())?;
    let stage = AcpStage::stage()?;
    let prompt_text = if turn.system.is_empty() {
        turn.content_line.clone()
    } else {
        format!("{}\n\n{}", turn.system, turn.content_line)
    };
    let mut child = std::process::Command::new(&binary)
        .args([
            "--config",
            &stage.cfg.join("config.json").to_string_lossy(),
            "acp",
            "--model",
            &turn.native_model,
        ])
        .current_dir(&stage.work)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .envs(setup::child_env())
        // A turn must never pop a browser out of a stale login.
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
        .map_err(|_| setup::INSTALL_HINT.to_string())?;

    // stderr → last ~20 lines, redacted, for error detail only.
    let stderr_tail: Arc<Mutex<std::collections::VecDeque<String>>> =
        Arc::new(Mutex::new(std::collections::VecDeque::new()));
    if let Some(err) = child.stderr.take() {
        let tail = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                if let Ok(mut t) = tail.lock() {
                    if t.len() >= 20 {
                        t.pop_front();
                    }
                    t.push_back(redact(&line));
                }
            }
        });
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "native stdout unavailable".to_string())?;
    let (tx, rx) = std::sync::mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(line)
                && tx.send(v).is_err()
            {
                break;
            }
        }
    });
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "native stdin unavailable".to_string())?;

    let deadline = Instant::now() + timeout;
    let mut rpc = Acp {
        stdin: &mut stdin,
        rx: &rx,
        next_id: 0,
        session_id: String::new(),
        text: String::new(),
        thought: String::new(),
        redirected: Vec::new(),
        usage: Usage::default(),
        names: turn.names.clone(),
        stderr_tail: &stderr_tail,
    };

    let outcome = drive(&mut rpc, &stage.work, &prompt_text, deadline);

    // Always: session/delete best effort, close stdin, brief wait, kill.
    if !rpc.session_id.is_empty() {
        rpc.send_request(
            "session/delete",
            json!({"sessionId": rpc.session_id.clone()}),
        )
        .ok();
        let _ = rpc.recv_response(Duration::from_secs(10), deadline);
    }
    drop(child.stdin.take());
    // Give the child up to 2s to exit after stdin closes, then kill.
    let wait_deadline = Instant::now() + Duration::from_secs(2);
    while child.try_wait().ok().flatten().is_none() && Instant::now() < wait_deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    outcome
}

struct AcpStage {
    work: std::path::PathBuf,
    cfg: std::path::PathBuf,
    _work_dir: tempfile::TempDir,
    _cfg_dir: tempfile::TempDir,
}

impl AcpStage {
    fn stage() -> Result<Self, String> {
        let work_dir = tempfile::Builder::new()
            .prefix("devin-sub-work-")
            .tempdir()
            .map_err(|e| format!("staging dir: {e}"))?;
        let cfg_dir = tempfile::Builder::new()
            .prefix("devin-sub-cfg-")
            .tempdir()
            .map_err(|e| format!("staging dir: {e}"))?;
        let work = work_dir.path().to_path_buf();
        let cfg = cfg_dir.path().to_path_buf();
        // Project-level deny: verified to actually block native read/exec
        // (the user-config deny alone does not).
        let proj = work.join(".devin");
        std::fs::create_dir_all(&proj).map_err(|e| format!("staging dir: {e}"))?;
        std::fs::write(
            proj.join("config.json"),
            json!({"permissions": {"deny": DENY_RULES}}).to_string(),
        )
        .map_err(|e| format!("staging config: {e}"))?;
        // User config: no MCP servers, rules, skills, subagents, auto-update
        // or notifications — but credentials still load.
        std::fs::write(
            cfg.join("config.json"),
            json!({
                "version": 1,
                "subagents_enabled": false,
                "auto_update": false,
                "notify": "never",
                "read_config_from": {
                    "agents_standard": false, "cursor": false, "windsurf": false,
                    "claude": false, "copilot": false, "opencode": false, "zed": false,
                },
                "permissions": {"deny": DENY_RULES, "allow": [], "ask": []},
            })
            .to_string(),
        )
        .map_err(|e| format!("staging config: {e}"))?;
        Ok(Self {
            work,
            cfg,
            _work_dir: work_dir,
            _cfg_dir: cfg_dir,
        })
    }
}

/// Per-turn ACP connection state.
struct Acp<'a> {
    stdin: &'a mut std::process::ChildStdin,
    rx: &'a std::sync::mpsc::Receiver<Value>,
    next_id: u64,
    session_id: String,
    text: String,
    thought: String,
    /// Native tool_call notifications redirected onto bash (name, args-json).
    redirected: Vec<(String, String)>,
    usage: Usage,
    names: Vec<String>,
    stderr_tail: &'a Arc<Mutex<std::collections::VecDeque<String>>>,
}

impl Acp<'_> {
    fn send(&mut self, v: &Value) -> Result<(), String> {
        let line = serde_json::to_string(v).map_err(|e| format!("frame encode: {e}"))? + "\n";
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.flush())
            .map_err(|_| "native stdin closed".to_string())
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<u64, String> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok(id)
    }

    fn answer_request(&mut self, id: &Value, result: Value) -> Result<(), String> {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    fn refuse_request(&mut self, id: &Value) -> Result<(), String> {
        self.send(&json!({"jsonrpc": "2.0", "id": id,
            "error": {"code": -32601, "message": "method not found"}}))
    }

    /// One server→client request: permissions always cancel, anything else
    /// is a JSON-RPC method-not-found (never a hang).
    fn on_request(&mut self, line: &Value) -> Result<(), String> {
        let id = line.get("id").cloned().unwrap_or(Value::Null);
        match line.get("method").and_then(Value::as_str) {
            Some("session/request_permission") => {
                self.answer_request(&id, json!({"outcome": {"outcome": "cancelled"}}))
            }
            _ => self.refuse_request(&id),
        }
    }

    fn on_notification(&mut self, line: &Value) {
        if line.get("method").and_then(Value::as_str) != Some("session/update") {
            return;
        }
        let update = &line["params"]["update"];
        match update.get("sessionUpdate").and_then(Value::as_str) {
            Some("agent_message_chunk") => {
                if let Some(t) = update.pointer("/content/text").and_then(Value::as_str) {
                    self.text.push_str(t);
                }
            }
            Some("agent_thought_chunk") => {
                if let Some(t) = update.pointer("/content/text").and_then(Value::as_str) {
                    self.thought.push_str(t);
                }
            }
            Some("tool_call") => {
                // First eligible redirect wins and ends the turn: the model
                // reached for a native tool instead of the text funnel.
                if self.redirected.is_empty()
                    && let Some(args) =
                        redirect_call(update, self.names.iter().any(|n| n == "bash"))
                {
                    self.redirected.push(("bash".to_string(), args));
                    let _ = self.send(&json!({"jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": self.session_id}}));
                }
            }
            Some("usage_update") => {
                let meta = &update["_meta"];
                let get = |k: &str| {
                    meta.get("cognition.ai")
                        .and_then(|c| c.get(k))
                        .or_else(|| meta.get(k))
                        .or_else(|| update.get(k))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize
                };
                self.usage.input_tokens = self.usage.input_tokens.max(get("inputTokens"));
                self.usage.output_tokens = self.usage.output_tokens.max(get("outputTokens"));
            }
            _ => {}
        }
    }

    /// Wait for the response to `want`: dispatching notifications and
    /// server requests as they arrive. `want == 0` waits for the prompt
    /// result specifically by method shape (any response while prompting).
    fn recv_response(&mut self, window: Duration, deadline: Instant) -> Result<Value, String> {
        let until = (Instant::now() + window).min(deadline);
        loop {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("Devin request timed out".into());
            }
            match self.rx.recv_timeout(left) {
                Ok(line) => {
                    if line.get("method").is_some() && line.get("id").is_some() {
                        self.on_request(&line)?;
                    } else if line.get("method").is_some() {
                        self.on_notification(&line);
                    } else if line.get("id").is_some() {
                        return Ok(line);
                    }
                }
                Err(_) => return Err("Devin request timed out".into()),
            }
        }
    }

    /// Read until the response to request `id` arrives (or a hard error).
    fn call(&mut self, method: &str, params: Value, window: Duration, deadline: Instant) -> Result<Value, String> {
        let id = self.send_request(method, params)?;
        loop {
            let line = self.recv_response(window, deadline)?;
            if line.get("id") == Some(&json!(id)) {
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, self.stderr_tail));
                }
                return Ok(line.get("result").cloned().unwrap_or(Value::Null));
            }
            // A response to another in-flight request: we send one at a
            // time, so anything else is protocol noise — keep waiting.
        }
    }

    /// Drain notifications until the `session/prompt` result arrives.
    fn prompt_result(&mut self, id: u64, deadline: Instant) -> Result<Value, String> {
        loop {
            let line = self.recv_response(Duration::from_secs(600), deadline)?;
            if line.get("id") == Some(&json!(id)) {
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, self.stderr_tail));
                }
                return Ok(line.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
}

fn drive(
    rpc: &mut Acp,
    work: &std::path::Path,
    prompt_text: &str,
    deadline: Instant,
) -> Result<TurnResult, String> {
    rpc.call(
        "initialize",
        json!({"protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false},
                "terminal": false}}),
        Duration::from_secs(60),
        deadline,
    )?;
    let new = rpc.call(
        "session/new",
        json!({"cwd": work.to_string_lossy(), "mcpServers": []}),
        Duration::from_secs(60),
        deadline,
    )?;
    let session_id = new
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    rpc.session_id = session_id.clone();
    let prompt_id = rpc.send_request(
        "session/prompt",
        json!({"sessionId": session_id,
            "prompt": [{"type": "text", "text": prompt_text}]}),
    )?;
    let result = rpc.prompt_result(prompt_id, deadline)?;
    let stop_reason = result
        .get("stopReason")
        .and_then(Value::as_str)
        .unwrap_or("");
    if let Some(u) = result.get("usage") {
        let get = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
        // inputTokens already includes cached tokens (verified: total =
        // input + output, cachedRead is a subset).
        rpc.usage.input_tokens = rpc.usage.input_tokens.max(get("inputTokens"));
        rpc.usage.output_tokens = rpc.usage.output_tokens.max(get("outputTokens"));
        rpc.usage.cached_tokens = get("cachedReadTokens");
    }
    match stop_reason {
        "refusal" => {
            return Err("Devin refused this request under its usage policy".to_string());
        }
        "end_turn" | "cancelled" | "max_tokens" | "max_turn_requests" | "" => {}
        other => {
            let _ = other;
        }
    }
    finish(rpc, stop_reason)
}

fn finish(rpc: &mut Acp, stop_reason: &str) -> Result<TurnResult, String> {
    let turn_id = rand_hex(8);
    // Redirected native calls never mix with funnel calls: redirect wins.
    let mut text = rpc.text.trim().to_string();
    let calls: Vec<(String, String)> = if !rpc.redirected.is_empty() {
        rpc.redirected.clone()
    } else {
        match parse_calls_block(&rpc.text, &rpc.names) {
            Some((before, calls)) => {
                text = before.trim().to_string();
                calls
            }
            None => Vec::new(),
        }
    };
    let calls: Vec<(String, String, String)> = calls
        .into_iter()
        .enumerate()
        .map(|(i, (name, args))| (format!("call_{turn_id}_{}", i + 1), name, args))
        .collect();
    let stop = match stop_reason {
        "max_tokens" => "incomplete:max_output_tokens".to_string(),
        _ if !calls.is_empty() => "tool_use".to_string(),
        _ => "completed".to_string(),
    };
    Ok(TurnResult {
        text,
        thought: rpc.thought.trim().to_string(),
        calls,
        usage: rpc.usage.clone(),
        stop,
    })
}

/// Map a JSON-RPC/CLI failure onto the host-facing error classes.
fn map_rpc_error(err: &Value, stderr: &Arc<Mutex<std::collections::VecDeque<String>>>) -> String {
    let mut detail = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if let Ok(tail) = stderr.lock() {
        let tail = tail.iter().cloned().collect::<Vec<_>>().join("; ");
        if !tail.is_empty() {
            detail = format!("{detail} {tail}");
        }
    }
    let detail = redact(&detail);
    let low = detail.to_lowercase();
    if low.contains("429") || low.contains("rate limit") || low.contains("quota") {
        return format!("Devin quota exhausted (native: {detail})");
    }
    if low.contains("not logged in") || low.contains("authenticate") || low.contains("unauthorized") {
        return setup::LOGIN_HINT.to_string();
    }
    if detail.trim().is_empty() {
        "native request failed".to_string()
    } else {
        format!("native request failed: {detail}")
    }
}

/// Redact token|secret|key|password-looking values from a detail string.
fn redact(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for word in s.split_whitespace() {
        let low = word.to_lowercase();
        let secretish = ["token", "secret", "key", "password"]
            .iter()
            .any(|k| low.contains(k) && (low.contains('=') || low.ends_with(':')));
        if secretish {
            let mut it = word.splitn(2, ['=', ':']);
            out.push_str(it.next().unwrap_or(word));
            out.push_str(if word.contains('=') { "=<redacted>" } else { ": <redacted>" });
        } else {
            out.push_str(word);
        }
        out.push(' ');
    }
    out.trim_end().to_string()
}

/// Fold a completed turn into a Responses SSE stream.
pub fn fold_result(
    r: &TurnResult,
    names: &[String],
) -> Result<Vec<u8>, String> {
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
        "incomplete:max_output_tokens" => (
            "incomplete",
            json!({"reason": "max_output_tokens"}),
        ),
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

fn rand_hex(n: usize) -> String {
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
