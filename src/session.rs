//! A pooled `devin acp` session: one live child process + ACP session that
//! answers consecutive host turns when each request's history is a strict
//! continuation of the one the session last answered.
//!
//! Devin's prompt cache is effectively per-session: a reused session
//! already holds the whole earlier transcript upstream (measured
//! ~99.8% cached on a repeat prompt), so a continuation turn prompts with
//! only the delta — tool results and new user input. Anything else (new
//! conversation, changed system text, diverged prefix) spawns a fresh
//! `devin acp` + `session/new`.
//!
//! Pool rules: at most `MAX_LIVE` live children; a session leaves the pool
//! for the duration of its turn (exclusive use) and returns on success —
//! including a client-disconnected turn whose prompt still settled: the
//! reply was never delivered (the relay buffers a whole turn), so the
//! session records an empty echo and the next request continues it with
//! only its own tail. Errors, timeouts and disconnects that fail to
//! settle close it instead — a session is never pooled in a state the
//! next turn can't trust. Sessions idle past `IDLE_TTL` or whose child
//! exited are reaped on every pool access and by a 60s background sweep;
//! [`shutdown`] closes everything.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::chat::{
    PreparedTurn, TurnResult, Usage, continuation, parse_calls_block, rand_hex, redirect_call,
    usage_from,
};
use crate::setup;

/// Pool bound: at most this many live `devin acp` children.
const MAX_LIVE: usize = 4;
/// A pooled session idle this long is closed on the next pool access.
const IDLE_TTL: Duration = Duration::from_secs(15 * 60);
/// How often an ACP wait wakes to check for a relay client disconnect.
const CANCEL_POLL: Duration = Duration::from_secs(1);
/// Internal error when the relay client went away mid-turn.
const CLIENT_GONE: &str = "client disconnected";

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

/// Process-wide staging: one scratch cwd + generated user config shared by
/// every pooled child, created lazily on first spawn and removed by
/// [`shutdown`]. Still random `tempfile` dirs — nothing is planted in a
/// shared /tmp path.
struct AcpStage {
    work: std::path::PathBuf,
    cfg: std::path::PathBuf,
    _work_dir: tempfile::TempDir,
    _cfg_dir: tempfile::TempDir,
}

static STAGE: OnceLock<AcpStage> = OnceLock::new();

impl AcpStage {
    fn make() -> Result<Self, String> {
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

    fn remove(&self) {
        let _ = std::fs::remove_dir_all(&self.work);
        let _ = std::fs::remove_dir_all(&self.cfg);
    }
}

/// The one stage every child in this sidecar process uses.
fn stage() -> Result<&'static AcpStage, String> {
    if let Some(s) = STAGE.get() {
        return Ok(s);
    }
    let made = AcpStage::make()?;
    Ok(STAGE.get_or_init(|| made))
}

/// One live `devin acp` child + ACP session, with the replay state needed
/// to recognize and answer its strict continuation turns.
pub struct LiveSession {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    next_id: u64,
    pub(crate) session_id: String,
    pub(crate) native_model: String,
    /// Funnel system text (instructions + contract incl. tool specs) the
    /// first prompt carried; a reused session only matches a request whose
    /// prepared system is identical.
    system: String,
    /// The exact `input` array of the last request this session answered.
    absorbed: Vec<Value>,
    /// Call ids emitted in that answer, in order.
    reply_call_ids: Vec<String>,
    /// The (trimmed) text emitted in that answer.
    reply_text: String,
    last_used: Instant,
    // Per-prompt state, reset in `send_prompt`.
    text: String,
    thought: String,
    /// Native tool_call notifications redirected onto bash (name, args-json).
    redirected: Vec<(String, String)>,
    /// Last `usage_update` snapshot for the in-flight prompt.
    usage: Usage,
    names: Vec<String>,
}

/// The process-wide session pool.
static POOL: OnceLock<Mutex<Vec<LiveSession>>> = OnceLock::new();

fn locked_pool() -> MutexGuard<'static, Vec<LiveSession>> {
    let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
    // Tiny reaper: closes idle/dead sessions even when no turn is running.
    static REAPER: Once = Once::new();
    REAPER.call_once(|| {
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(Duration::from_secs(60));
                let dead = {
                    let Some(pool) = POOL.get() else { continue };
                    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
                    reap(&mut pool)
                };
                for s in dead {
                    s.close();
                }
            }
        });
    });
    pool.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pull dead/idle sessions out of a locked pool (caller closes them after
/// releasing the lock — `close` can block a few seconds on `session/delete`).
fn reap(pool: &mut Vec<LiveSession>) -> Vec<LiveSession> {
    let mut dead = Vec::new();
    let mut i = 0;
    while i < pool.len() {
        let stale =
            pool[i].last_used.elapsed() > IDLE_TTL || !matches!(pool[i].child.try_wait(), Ok(None));
        if stale {
            dead.push(pool.remove(i));
        } else {
            i += 1;
        }
    }
    dead
}

/// Take the first pooled session this turn is a strict continuation of
/// (same native model, same system text, `continuation` yields a delta).
/// Returns the session plus the delta to prompt with; on a miss the
/// second tuple item is the deepest reason any candidate reached
/// (`no_session` when the pool was empty).
pub fn take(turn: &PreparedTurn) -> (Option<(LiveSession, String)>, &'static str) {
    let mut reason = "no_session";
    let (found, dead) = {
        let mut pool = locked_pool();
        let dead = reap(&mut pool);
        let mut depth = 0;
        let mut found = None;
        for (i, s) in pool.iter().enumerate() {
            let (miss, d) = if s.native_model != turn.native_model {
                ("model", 1)
            } else if s.system != turn.system {
                ("system", 2)
            } else {
                match continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &turn.input) {
                    Ok(delta) => {
                        found = Some((i, delta));
                        break;
                    }
                    Err(m) => (
                        m,
                        match m {
                            "prefix" => 3,
                            "echo" => 4,
                            _ => 5,
                        },
                    ),
                }
            };
            if d > depth {
                depth = d;
                reason = miss;
            }
        }
        (found.map(|(i, delta)| (pool.remove(i), delta)), dead)
    };
    for s in dead {
        s.close();
    }
    (found, reason)
}

/// Return a session to the pool after a successful turn, evicting the
/// least-recently-used session past `MAX_LIVE`.
pub fn give_back(s: LiveSession) {
    let mut dead = {
        let mut pool = locked_pool();
        let mut dead = reap(&mut pool);
        pool.push(s);
        if pool.len() > MAX_LIVE {
            let lru = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            dead.push(pool.remove(lru));
        }
        dead
    };
    for s in dead.drain(..) {
        s.close();
    }
}

/// Close every pooled session and remove the process-wide stage dirs.
/// Called on `plugin/shutdown` and when the host's stdin goes away.
pub fn shutdown() {
    let dead = {
        let mut pool = locked_pool();
        std::mem::take(&mut *pool)
    };
    for s in dead {
        s.close();
    }
    if let Some(stage) = STAGE.get() {
        stage.remove();
    }
}

/// Spawn a fresh `devin acp` child and drive initialize + session/new.
/// `cancel` aborts the handshake when the relay client is already gone.
pub fn spawn(
    turn: &PreparedTurn,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<LiveSession, String> {
    let binary = setup::resolve_command().ok_or_else(|| setup::INSTALL_HINT.to_string())?;
    let stage = stage()?;
    let mut child = Command::new(&binary)
        .args([
            "--config",
            &stage.cfg.join("config.json").to_string_lossy(),
            "acp",
            "--model",
            &turn.native_model,
        ])
        .current_dir(&stage.work)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(setup::child_env())
        // A turn must never pop a browser out of a stale login.
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
        .map_err(|_| setup::INSTALL_HINT.to_string())?;

    // stderr → last ~20 lines, redacted, for error detail only.
    let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
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
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "native stdin unavailable".to_string())?;

    let mut s = LiveSession {
        child,
        stdin,
        rx,
        stderr_tail,
        next_id: 0,
        session_id: String::new(),
        native_model: turn.native_model.clone(),
        system: turn.system.clone(),
        absorbed: Vec::new(),
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        last_used: Instant::now(),
        text: String::new(),
        thought: String::new(),
        redirected: Vec::new(),
        usage: Usage::default(),
        names: turn.names.clone(),
    };
    if let Err(e) = s.handshake(&stage.work, deadline, cancel) {
        s.close();
        return Err(e);
    }
    Ok(s)
}

impl LiveSession {
    /// initialize → session/new; leaves the session ready for prompts.
    fn handshake(
        &mut self,
        work: &std::path::Path,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        self.call(
            "initialize",
            json!({"protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false},
                    "terminal": false}}),
            Duration::from_secs(60),
            deadline,
            cancel,
        )?;
        let new = self.call(
            "session/new",
            json!({"cwd": work.to_string_lossy(), "mcpServers": []}),
            Duration::from_secs(60),
            deadline,
            cancel,
        )?;
        self.session_id = new
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok(())
    }

    /// Drain stale wire traffic, reset per-prompt state and send the
    /// `session/prompt` request. Late chunks from a previous prompt must
    /// not leak into this one: queued server requests are answered
    /// (permissions cancel, the rest refused), stale notifications and
    /// responses are dropped.
    pub(crate) fn send_prompt(&mut self, text: &str, names: &[String]) -> Result<u64, String> {
        while let Ok(v) = self.rx.try_recv() {
            if v.get("method").is_some() && v.get("id").is_some() {
                self.on_request(&v)?;
            }
        }
        self.text.clear();
        self.thought.clear();
        self.redirected.clear();
        self.usage = Usage::default();
        self.names = names.to_vec();
        self.send_request(
            "session/prompt",
            json!({"sessionId": self.session_id,
                "prompt": [{"type": "text", "text": text}]}),
        )
    }

    /// Wait out an accepted prompt and fold it into a TurnResult.
    pub(crate) fn await_prompt(
        &mut self,
        id: u64,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<TurnResult, String> {
        let (result, unseen) = self.prompt_result(id, deadline, cancel)?;
        // `result.usage` wins when present and non-zero; a cancelled turn
        // carries none, so the last usage_update snapshot stands in.
        if let Some(u) = result.get("usage").and_then(usage_from) {
            self.usage = u;
        }
        let stop_reason = result
            .get("stopReason")
            .and_then(Value::as_str)
            .unwrap_or("");
        if stop_reason == "refusal" {
            return Err("Devin refused this request under its usage policy".to_string());
        }
        let mut r = self.finish(stop_reason);
        r.unseen = unseen;
        Ok(r)
    }

    /// Record what this session just answered so the next request can be
    /// recognized as its strict continuation. An unseen answer (the relay
    /// client was already gone when the turn settled) records an empty
    /// echo: gray's history carries no assistant items for it, so the next
    /// request's echo zone is empty too.
    pub(crate) fn absorb(&mut self, turn: &PreparedTurn, r: &TurnResult) {
        self.system = turn.system.clone();
        self.absorbed = turn.input.clone();
        self.reply_call_ids = if r.unseen {
            Vec::new()
        } else {
            r.calls.iter().map(|(id, _, _)| id.clone()).collect()
        };
        self.reply_text = if r.unseen {
            String::new()
        } else {
            r.text.trim().to_string()
        };
        self.last_used = Instant::now();
    }

    /// Best-effort teardown: `session/delete` (≤5s), stdin closed, ≤2s
    /// grace, then kill.
    pub fn close(mut self) {
        if !self.session_id.is_empty()
            && self
                .send_request("session/delete", json!({"sessionId": self.session_id}))
                .is_ok()
        {
            let never = AtomicBool::new(false);
            let _ = self.recv_response(
                Duration::from_secs(5),
                Instant::now() + Duration::from_secs(5),
                &never,
            );
        }
        drop(self.stdin);
        let wait_deadline = Instant::now() + Duration::from_secs(2);
        while self.child.try_wait().ok().flatten().is_none() && Instant::now() < wait_deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

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
                // Keep the LAST snapshot: a later update carries the final
                // counters for this prompt.
                if let Some(u) = usage_from(update) {
                    self.usage = u;
                }
            }
            _ => {}
        }
    }

    /// Wait for the next response on the wire, dispatching notifications
    /// and server requests as they arrive. Wakes at least every
    /// `CANCEL_POLL` to check `cancel`: a relayed client disconnect must
    /// end the wait, never hang it.
    fn recv_response(
        &mut self,
        window: Duration,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Value, String> {
        let until = (Instant::now() + window).min(deadline);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(CLIENT_GONE.to_string());
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("Devin request timed out".into());
            }
            match self.rx.recv_timeout(left.min(CANCEL_POLL)) {
                Ok(line) => {
                    if line.get("method").is_some() && line.get("id").is_some() {
                        self.on_request(&line)?;
                    } else if line.get("method").is_some() {
                        self.on_notification(&line);
                    } else if line.get("id").is_some() {
                        return Ok(line);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("native stdout closed".into());
                }
            }
        }
    }

    /// Read until the response to request `id` arrives (or a hard error).
    fn call(
        &mut self,
        method: &str,
        params: Value,
        window: Duration,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Value, String> {
        let id = self.send_request(method, params)?;
        loop {
            let line = self.recv_response(window, deadline, cancel)?;
            if line.get("id") == Some(&json!(id)) {
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, &self.stderr_tail));
                }
                return Ok(line.get("result").cloned().unwrap_or(Value::Null));
            }
            // A response to another in-flight request: we send one at a
            // time, so anything else is protocol noise — keep waiting.
        }
    }

    /// Drain notifications until the `session/prompt` result arrives,
    /// plus an `unseen` flag. On client disconnect: `session/cancel` once,
    /// then ≤10s for the prompt result to settle. A settled result returns
    /// `unseen: true` — the session state is then known (the turn is over
    /// and its reply never reached the client), so the caller can pool
    /// it; only a grace timeout errors and costs the session.
    fn prompt_result(
        &mut self,
        id: u64,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(Value, bool), String> {
        loop {
            let line = match self.recv_response(Duration::from_secs(600), deadline, cancel) {
                Ok(line) => line,
                Err(e) => {
                    if e == CLIENT_GONE {
                        let _ = self.send(&json!({"jsonrpc": "2.0",
                            "method": "session/cancel",
                            "params": {"sessionId": self.session_id}}));
                        let grace = Instant::now() + Duration::from_secs(10);
                        let never = AtomicBool::new(false);
                        while Instant::now() < grace {
                            match self.recv_response(Duration::from_secs(10), grace, &never) {
                                Ok(line) if line.get("id") == Some(&json!(id)) => {
                                    if let Some(err) = line.get("error") {
                                        return Err(map_rpc_error(err, &self.stderr_tail));
                                    }
                                    return Ok((
                                        line.get("result").cloned().unwrap_or(Value::Null),
                                        true,
                                    ));
                                }
                                Ok(_) => {}
                                Err(_) => break,
                            }
                        }
                    }
                    return Err(e);
                }
            };
            if line.get("id") == Some(&json!(id)) {
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, &self.stderr_tail));
                }
                return Ok((line.get("result").cloned().unwrap_or(Value::Null), false));
            }
        }
    }

    /// Fold the accumulated reply into a TurnResult.
    fn finish(&mut self, stop_reason: &str) -> TurnResult {
        let turn_id = rand_hex(8);
        // Redirected native calls never mix with funnel calls: redirect wins.
        let mut text = self.text.trim().to_string();
        let calls: Vec<(String, String)> = if !self.redirected.is_empty() {
            self.redirected.clone()
        } else {
            match parse_calls_block(&self.text, &self.names) {
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
        TurnResult {
            text,
            thought: self.thought.trim().to_string(),
            calls,
            usage: self.usage.clone(),
            stop,
            unseen: false,
        }
    }
}

/// Map a JSON-RPC/CLI failure onto the host-facing error classes.
fn map_rpc_error(err: &Value, stderr: &Arc<Mutex<VecDeque<String>>>) -> String {
    let mut detail = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if detail.trim().is_empty()
        && let Ok(tail) = stderr.lock()
        && let Some(last) = tail.iter().rev().find(|line| !line.trim().is_empty())
    {
        detail = last.clone();
    }
    let detail = redact(&detail);
    let low = detail.to_lowercase();
    if low.contains("quota") || low.contains("reached free model rate limit") {
        return format!("Devin usage limit reached: {detail}");
    }
    if low.contains("429") || low.contains("rate limit") {
        return format!("Devin rate limited: {detail}");
    }
    if low.contains("not logged in") || low.contains("authenticate") || low.contains("unauthorized")
    {
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
            out.push_str(if word.contains('=') {
                "=<redacted>"
            } else {
                ": <redacted>"
            });
        } else {
            out.push_str(word);
        }
        out.push(' ');
    }
    out.trim_end().to_string()
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
