use super::*;
use serde_json::json;

#[test]
fn web_search_is_denied_only_off_the_advertised_surface() {
    // deny_search=true: the child config kills native web_search — the
    // model gets Devin's own denial, not a funnel-made native__ name.
    let stage = AcpStage::make(true).unwrap();
    let proj: Value = serde_json::from_str(
        &std::fs::read_to_string(stage.work.join(".devin/config.json")).unwrap(),
    )
    .unwrap();
    let user: Value =
        serde_json::from_str(&std::fs::read_to_string(stage.cfg.join("config.json")).unwrap())
            .unwrap();
    let deny = proj["permissions"]["deny"].as_array().unwrap();
    assert!(deny.iter().any(|r| r == "web_search"));
    let deny = user["permissions"]["deny"].as_array().unwrap();
    assert!(deny.iter().any(|r| r == "web_search"));

    // deny_search=false: host advertises web_search, funnel redirects it —
    // the upstream deny would race the redirect, so it stays off.
    let stage = AcpStage::make(false).unwrap();
    let proj: Value = serde_json::from_str(
        &std::fs::read_to_string(stage.work.join(".devin/config.json")).unwrap(),
    )
    .unwrap();
    let deny = proj["permissions"]["deny"].as_array().unwrap();
    assert!(!deny.iter().any(|r| r == "web_search"));
    let ask = proj["permissions"]["ask"].as_array().unwrap();
    assert!(ask.iter().any(|r| r == "exec"));
}

fn stderr(lines: &[&str]) -> Arc<Mutex<VecDeque<String>>> {
    Arc::new(Mutex::new(
        lines.iter().map(|line| line.to_string()).collect(),
    ))
}

#[test]
fn rpc_message_does_not_include_stderr_retry_history() {
    let detail = map_rpc_error(
        &json!({"message": "Reached free model rate limit. Your limit will reset in 6 hours (at 21:01 UTC)."}),
        &stderr(&[
            "WARN attempt=1 error=Inference(ServerError)",
            "WARN attempt=2 retrying",
        ]),
    );
    assert!(detail.starts_with("Devin usage limit reached:"), "{detail}");
    assert!(detail.contains("21:01 UTC"), "{detail}");
    assert!(!detail.contains("WARN"), "{detail}");
    assert!(!detail.contains("attempt"), "{detail}");
}

#[test]
fn transient_rate_limit_is_not_called_exhausted_quota() {
    let detail = map_rpc_error(
        &json!({"message": "status 429: rate limit, try again shortly"}),
        &stderr(&[]),
    );
    assert!(detail.starts_with("Devin rate limited:"), "{detail}");
    assert!(!detail.contains("quota"), "{detail}");
}

#[test]
fn missing_rpc_message_uses_only_the_latest_stderr_line() {
    let detail = map_rpc_error(
        &json!({}),
        &stderr(&["old retry noise", "request failed", "  "]),
    );
    assert_eq!(detail, "native request failed: request failed");
}

#[test]
fn fallback_stderr_remains_redacted() {
    let detail = map_rpc_error(
        &json!({}),
        &stderr(&["request failed token=sensitive-placeholder"]),
    );
    assert!(!detail.contains("sensitive-placeholder"), "{detail}");
    assert!(detail.contains("<redacted>"), "{detail}");
}

/// A `sleep` child stands in for `devin acp`: alive with a pipeable
/// stdin and controllable clocks — no handshake needed to exercise pool
/// bookkeeping. Tests kill the child before returning.
fn stub_session() -> LiveSession {
    #[cfg(unix)]
    let mut child = Command::new("sleep")
        .arg("60")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("sleep");
    #[cfg(windows)]
    let mut child = Command::new("powershell")
        .args(["-NoProfile", "-Command", "Start-Sleep", "-Seconds", "60"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("powershell");
    let stdin = child.stdin.take().expect("stdin");
    let (_tx, rx) = std::sync::mpsc::channel::<Value>();
    LiveSession {
        child,
        stdin,
        rx,
        stderr_tail: stderr(&[]),
        limit_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        limit_polls: 0,
        next_id: 0,
        current_prompt_id: None,
        session_id: "s-test".to_string(),
        native_model: "model".to_string(),
        system: "sys".to_string(),
        stage_work: std::path::PathBuf::new(),
        account: String::new(),
        absorbed: Vec::new(),
        lead_deltas_since_nudge: 0,
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        last_reply: None,
        last_used: Instant::now(),
        last_prompt_at: None,
        keepalive_turns: 0,
        prompt_in_flight: false,
        text: String::new(),
        text_boundary: false,
        thought: String::new(),
        redirected: Vec::new(),
        dropped_siblings: 0,
        prior_dropped: 0,
        seen_calls: std::collections::HashSet::new(),
        sidekick_calls: std::collections::HashMap::new(),
        active_sidekick: None,
        unhandled_sidekick: std::collections::HashSet::new(),
        terminals: std::collections::HashMap::new(),
        parked: None,
        seen_update_kinds: std::collections::HashSet::new(),
        pending_perms: Vec::new(),
        pending_lead: Vec::new(),
        batch_quiet: None,
        batch_flushed: false,
        guard: crate::guard::TurnGuard::default(),
        prior_cut: false,
        usage: Usage::default(),
        names: Vec::new(),
        keepalive_prompt: false,
        loading: false,
        gray_session: String::new(),
        deny_search: false,
        rewarmed: false,
    }
}

#[test]
fn keepalive_turns_do_not_break_continuation_matching() {
    let mut s = stub_session();
    // One real turn absorbed: history [user "hi"] answered "answer".
    let turn = PreparedTurn {
        system: "sys".to_string(),
        content_line: "hi".to_string(),
        input: vec![json!({"role": "user", "content": "hi"})],
        names: Vec::new(),
        native_model: "model".to_string(),
        gray_session: String::new(),
        account: None,
    };
    let r = TurnResult {
        text: "answer".to_string(),
        thought: String::new(),
        calls: Vec::new(),
        usage: Usage::default(),
        stop: "completed".to_string(),
        unseen: false,
    };
    s.absorb(&turn, &r);
    // The next host request replays the same history, our echoed answer,
    // and a new user item — the strict continuation of what we absorbed.
    let mut input = turn.input.clone();
    input.push(json!({"role": "assistant", "content": "answer"}));
    input.push(json!({"role": "user", "content": "more"}));
    let before = continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &input)
        .expect("continuation before keepalive");
    // Two injected `.` turns land in the child's transcript only: gray's
    // matching state is untouched, so the delta is identical.
    s.absorb_keepalive();
    s.absorb_keepalive();
    let after = continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &input)
        .expect("continuation after keepalives");
    assert_eq!(before, after);
    assert_eq!(s.keepalive_turns, 2);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

#[test]
fn keepalive_updates_upstream_clock_not_user_clock() {
    let mut s = stub_session();
    let user_clock = Instant::now() - Duration::from_secs(300);
    s.last_used = user_clock;
    s.absorb_keepalive();
    assert_eq!(s.last_used, user_clock, "keepalive bumped the idle clock");
    let upstream = s
        .last_prompt_at
        .expect("keepalive must stamp upstream contact");
    assert!(upstream.elapsed() < Duration::from_secs(5));
    assert_eq!(s.keepalive_turns, 1);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

#[test]
fn keepalives_do_not_extend_idle_ttl() {
    let mut s = stub_session();
    // Fresh upstream contact (recently warmed) but the user went idle
    // past IDLE_TTL: the reaper measures user activity only.
    s.absorb_keepalive();
    s.last_used = Instant::now() - IDLE_TTL - Duration::from_secs(1);
    let mut pool = vec![s];
    let dead = reap(&mut pool);
    assert_eq!(dead.len(), 1, "warmed-but-idle session survived the reaper");
    assert!(pool.is_empty());
    for s in dead {
        s.close();
    }
}

#[test]
fn a_turn_waits_for_its_session_to_return_from_a_keepalive() {
    // Unique model: the pool is process-wide and tests run in parallel.
    let model = "model-keepalive-wait";
    let turn = |model: &str, input: Vec<Value>| PreparedTurn {
        system: "sys".to_string(),
        content_line: String::new(),
        input,
        names: Vec::new(),
        native_model: model.to_string(),
        gray_session: String::new(),
        account: None,
    };
    let history = vec![json!({"role": "user", "content": "hi"})];
    let first = turn(model, history.clone());
    let mut s = stub_session();
    s.session_id = "s-keepalive-wait".to_string();
    s.native_model = model.to_string();
    s.absorb(
        &first,
        &TurnResult {
            text: "answer".to_string(),
            thought: String::new(),
            calls: Vec::new(),
            usage: Usage::default(),
            stop: "completed".to_string(),
            unseen: false,
        },
    );
    let mut input = history;
    input.push(json!({"role": "assistant", "content": "answer"}));
    input.push(json!({"role": "user", "content": "more"}));
    let next = turn(model, input.clone());
    let never = AtomicBool::new(false);

    // Out for a keepalive: the turn must not miss straight to a fresh
    // spawn, it waits and gets the same session back.
    locked_pool().warming.push(Warming {
        session_id: s.session_id.clone(),
        native_model: model.to_string(),
        system: "sys".to_string(),
        account: String::new(),
    });
    let back = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        give_back(s);
    });
    let started = Instant::now();
    let (claimed, _) = take(&next, &never);
    back.join().unwrap();
    let (mut s, delta) = claimed.expect("turn missed while its session was warming");
    assert!(started.elapsed() >= Duration::from_millis(250));
    assert_eq!(s.session_id, "s-keepalive-wait");
    assert_eq!(delta, "[User]\nmore");
    assert!(!locked_pool().warming_for(&next));

    // Nothing warming for this turn: a miss returns at once.
    let other = turn("model-keepalive-none", input);
    let started = Instant::now();
    let (claimed, reason) = take(&other, &never);
    assert!(claimed.is_none());
    assert_ne!(reason, "keepalive_busy");
    assert!(started.elapsed() < Duration::from_millis(250));

    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// The recache regression: a request landing while its conversation's
/// session is checked out mid-prompt used to fall straight to a cold
/// `session/new` — and its `session/load` on the same record was a
/// guaranteed `session_locked`. The borrow mark now makes `take` wait
/// for `give_back` and continue warm; a different conversation sharing
/// model+system still returns immediately.
#[test]
fn take_waits_for_a_checked_out_session() {
    let model = "model-borrowed-wait";
    let history = vec![json!({"role": "user", "content": "hi"})];
    let mut first = turn(model, history.clone());
    first.gray_session = "conv-a".to_string();
    let mut s = stub_session();
    s.session_id = "s-borrowed-wait".to_string();
    s.native_model = model.to_string();
    s.absorb(&first, &answered("answer"));
    give_back(s);

    // Turn A of the conversation checks the session out.
    let mut in_a = history;
    in_a.push(json!({"role": "assistant", "content": "answer"}));
    in_a.push(json!({"role": "user", "content": "turn-a"}));
    let mut ta = turn(model, in_a.clone());
    ta.gray_session = "conv-a".to_string();
    let never = AtomicBool::new(false);
    let (claimed, _) = take(&ta, &never);
    let mut s = claimed.expect("live session claimable").0;
    assert!(locked_pool().borrowed_for(&ta));

    // A different conversation sharing model+system does not wait on it.
    let mut other = turn(model, in_a);
    other.gray_session = "conv-b".to_string();
    let started = Instant::now();
    let (claimed, reason) = take(&other, &never);
    assert!(claimed.is_none());
    assert_eq!(reason, "no_session");
    assert!(started.elapsed() < Duration::from_millis(250));

    // Same conversation, still checked out: the request waits for
    // give_back and reuses the returned session warm.
    s.absorb(&ta, &answered("answer-a"));
    let mut in_b = ta.input.clone();
    in_b.push(json!({"role": "assistant", "content": "answer-a"}));
    in_b.push(json!({"role": "user", "content": "more"}));
    let mut tb = turn(model, in_b);
    tb.gray_session = "conv-a".to_string();
    let back = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        give_back(s);
    });
    let started = Instant::now();
    let (claimed, _) = take(&tb, &never);
    back.join().unwrap();
    let (s, delta) = claimed.expect("turn missed while its session was checked out");
    assert!(started.elapsed() >= Duration::from_millis(250));
    assert_eq!(s.session_id, "s-borrowed-wait");
    assert_eq!(delta, "[User]\nmore");
    assert!(locked_pool().borrowed_for(&tb));

    // Closing the borrowed session releases the mark for its
    // conversation (give_back does the same on the warm path).
    s.close();
    assert!(!locked_pool().borrowed_for(&tb));
}

/// A stub `devin acp` child: JSON-RPC over stdio, answers initialize /
/// session/new / session/load / session/delete and replies to
/// session/prompt with one message chunk + usage. Every request is
/// appended to @LOG@ so the test can assert exactly which methods ran.
#[cfg(unix)]
const STUB_ACP: &str = r##"#!/usr/bin/env python3
import sys, json, os
LOG = "@LOG@"
def log(method, extra=None):
    with open(LOG, "a") as f:
        f.write(json.dumps(dict({"method": method}, **(extra or {}))) + "\n")
sid = "stub-%d" % os.getpid()
for line in sys.stdin:
    try:
        m = json.loads(line)
    except Exception:
        continue
    method, rid = m.get("method"), m.get("id")
    params = m.get("params") or {}
    if method == "session/cancel" or rid is None:
        continue
    if method == "session/prompt":
        text = "".join(p.get("text", "") for p in params.get("prompt", []))
        log(method, {"sessionId": params.get("sessionId"), "text": text})
        u = {"sessionUpdate": "agent_message_chunk",
             "content": {"type": "text", "text": "stub reply"}}
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update",
                          "params": {"sessionId": params.get("sessionId"), "update": u}}))
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": {
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1000, "outputTokens": 10,
                      "cachedReadTokens": 900}}}))
    elif method == "initialize":
        log(method)
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result":
            {"protocolVersion": 1, "agentCapabilities": {"loadSession": True}}}))
    elif method == "session/new":
        log(method)
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": sid}}))
    elif method in ("session/load", "session/delete"):
        log(method, {"sessionId": params.get("sessionId")})
        print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": {}}))
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid,
                          "error": {"code": -32601, "message": "unknown"}}))
    sys.stdout.flush()
"##;

/// End-to-end against the stub child — real spawn, real JSON-RPC, real
/// pool — proving the cycle a unit stub can't: fresh `session/new`,
/// prompt, pool; strict reuse prompting only the delta; a keepalive on
/// the due session; idle reap detaching without `session/delete`; and a
/// later turn reattaching through `session/load`.
// The stub child is a python script run via its shebang: unix only.
#[cfg(unix)]
#[test]
fn e2e_stub_child_pool_cycle() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dir = std::env::temp_dir().join(format!("devin-sub-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("child.log");
    let stub_path = dir.join("devin-stub");
    std::fs::write(
        &stub_path,
        STUB_ACP.replace("@LOG@", &log_path.to_string_lossy()),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    // SAFETY: serialized by LOCK; no other test reads these vars.
    unsafe {
        std::env::set_var("GRAY_DEVIN_SUB_COMMAND", &stub_path);
        std::env::set_var("GRAY_HOME", dir.join("gray-home"));
    }
    let read_log = || -> Vec<Value> {
        std::fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    };
    let count = |method: &str| -> usize {
        read_log()
            .iter()
            .filter(|e| e.get("method").and_then(Value::as_str) == Some(method))
            .count()
    };

    let never = AtomicBool::new(false);
    let deadline = Instant::now() + Duration::from_secs(30);
    let model = "e2e-stub-model";

    // Fresh turn: spawn + handshake + real prompt over the wire.
    let mut t1 = turn(model, vec![json!({"role": "user", "content": "hi"})]);
    t1.gray_session = "conv-e2e".to_string();
    let mut s = spawn(&t1, deadline, &never).expect("spawn failed");
    let r = s
        .send_prompt("hi", &[], &[])
        .and_then(|id| s.await_prompt(id, deadline, &never))
        .expect("prompt failed");
    assert_eq!(r.text, "stub reply");
    s.absorb(&t1, &r);
    let sid = s.session_id.clone();
    give_back(s);
    assert_eq!(count("session/new"), 1);
    assert_eq!(count("session/prompt"), 1);

    // Strict continuation: the pooled child answers only the delta.
    let mut t2 = turn(
        model,
        vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "stub reply"}),
            json!({"role": "user", "content": "next"}),
        ],
    );
    t2.gray_session = "conv-e2e".to_string();
    let (claimed, reason) = take(&t2, &never);
    let (mut s, delta) = claimed.unwrap_or_else(|| panic!("continuation missed: {reason}"));
    assert_eq!(s.session_id, sid);
    assert_eq!(delta, "[User]\nnext");
    let r = s
        .send_prompt(&delta, &[], &[])
        .and_then(|id| s.await_prompt(id, deadline, &never))
        .expect("delta prompt failed");
    s.absorb(&t2, &r);
    give_back(s);
    assert_eq!(count("session/new"), 1, "reuse spawned a fresh child");
    assert_eq!(count("session/prompt"), 2);
    let last = read_log().pop().unwrap();
    assert_eq!(last["text"], "[User]\nnext");

    // Keepalive: staleness due → one marker prompt, no delete.
    {
        let mut pool = locked_pool();
        let s = pool
            .live
            .iter_mut()
            .find(|s| s.session_id == sid)
            .expect("pooled session gone");
        s.last_prompt_at = Some(Instant::now() - KEEPALIVE_AFTER - Duration::from_secs(1));
    }
    let mut s = take_keepalive().expect("due session not offered for keepalive");
    assert_eq!(s.session_id, sid);
    s.keepalive().expect("keepalive prompt failed");
    give_back(s);
    let last = read_log().pop().unwrap();
    assert_eq!(last["method"], "session/prompt");
    assert_eq!(last["text"], KEEPALIVE_TEXT);

    // Idle reap detaches: child exits, record survives, no delete.
    {
        let mut pool = locked_pool();
        let s = pool
            .live
            .iter_mut()
            .find(|s| s.session_id == sid)
            .expect("pooled session gone");
        s.last_used = Instant::now() - IDLE_TTL - Duration::from_secs(1);
    }
    // A take of an unrelated turn sweeps the idle session out.
    let other = turn(
        "e2e-unrelated",
        vec![json!({"role": "user", "content": "x"})],
    );
    let _ = take(&other, &never);
    assert!(
        !locked_pool().live.iter().any(|s| s.session_id == sid),
        "idle session was not reaped"
    );
    assert_eq!(count("session/delete"), 0, "reap deleted the session");
    assert_eq!(count("session/load"), 0);

    // Reattach: a new child `session/load`s the detached session and the
    // next turn prompts only the delta — not a fresh transcript.
    let mut t3 = turn(
        model,
        vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "stub reply"}),
            json!({"role": "user", "content": "next"}),
            json!({"role": "assistant", "content": "stub reply"}),
            json!({"role": "user", "content": "again"}),
        ],
    );
    t3.gray_session = "conv-e2e".to_string();
    let (mut s, delta) =
        reattach(&t3, deadline, &never).expect("reattach missed the detached session");
    assert_eq!(s.session_id, sid);
    assert_eq!(delta, "[User]\nagain");
    assert_eq!(count("session/load"), 1);
    assert_eq!(count("session/new"), 1, "reattach spawned session/new");
    let r = s
        .send_prompt(&delta, &[], &[])
        .and_then(|id| s.await_prompt(id, deadline, &never))
        .expect("reattached prompt failed");
    s.absorb(&t3, &r);
    s.close();
    assert_eq!(count("session/delete"), 1, "close() must still delete");

    unsafe {
        std::env::remove_var("GRAY_DEVIN_SUB_COMMAND");
        std::env::remove_var("GRAY_HOME");
    }
}

#[test]
fn rewritten_history_never_reuses_the_upstream_session() {
    let model = "model-rewritten-tail";
    let start = json!({"role": "user", "content": "hi"});
    let mut first = turn(
        model,
        vec![
            start.clone(),
            json!({"role": "assistant", "content": "original answer"}),
            json!({"role": "user", "content": "previous"}),
        ],
    );
    first.gray_session = "conv-rewrite".to_string();
    let mut s = stub_session();
    s.session_id = "s-rewrite".to_string();
    s.native_model = model.to_string();
    s.absorb(&first, &answered("answer"));
    give_back(s);

    let never = AtomicBool::new(false);
    let mut compacted = turn(
        model,
        vec![
            json!({"role": "user", "content": "[summary of earlier turns]"}),
            json!({"role": "assistant", "content": "kept reply"}),
            json!({"role": "user", "content": "continue"}),
        ],
    );
    compacted.gray_session = "conv-rewrite".to_string();
    let (claimed, _) = take(&compacted, &never);
    assert!(
        claimed.is_none(),
        "compaction reused the old upstream session"
    );

    let mut cut = turn(
        model,
        vec![
            start.clone(),
            json!({"role": "assistant", "content": "original answer"}),
        ],
    );
    cut.gray_session = "conv-rewrite".to_string();
    let (claimed, _) = take(&cut, &never);
    assert!(
        claimed.is_none(),
        "cut history reused the old upstream session"
    );

    let mut drift = turn(
        model,
        vec![
            start.clone(),
            json!({"role": "assistant", "content": "edited answer"}),
            json!({"role": "user", "content": "rewritten question"}),
            json!({"role": "assistant", "content": "kept reply"}),
            json!({"role": "user", "content": "another revised turn"}),
            json!({"role": "assistant", "content": "still in history"}),
            json!({"role": "user", "content": "continue"}),
        ],
    );
    drift.gray_session = "conv-rewrite".to_string();
    let (claimed, _) = take(&drift, &never);
    assert!(
        claimed.is_none(),
        "rewritten history reused the old upstream session"
    );

    let mut other = turn(
        model,
        vec![json!({"role": "user", "content": "unrelated start"})],
    );
    other.gray_session = "conv-other".to_string();
    let (claimed, _) = take(&other, &never);
    assert!(claimed.is_none());

    let mut next = turn(
        model,
        vec![
            start,
            json!({"role": "assistant", "content": "original answer"}),
            json!({"role": "user", "content": "previous"}),
            json!({"role": "assistant", "content": "answer"}),
            json!({"role": "user", "content": "continue"}),
        ],
    );
    next.gray_session = "conv-rewrite".to_string();
    let (claimed, reason) = take(&next, &never);
    let (s, delta) = claimed.unwrap_or_else(|| panic!("strict continuation missed: {reason}"));
    assert_eq!(s.session_id, "s-rewrite");
    assert_eq!(delta, "[User]\ncontinue");
    s.close();
}

#[test]
fn keepalive_due_bounds_warming() {
    let active = Instant::now();
    let stale = Some(Instant::now() - KEEPALIVE_AFTER - Duration::from_secs(1));
    // Never served a real prompt → never warmed.
    assert!(!keepalive_due(None, active, true));
    // Upstream contact still fresh → not due.
    assert!(!keepalive_due(Some(Instant::now()), active, true));
    // Stale cache + active user + live child → due.
    assert!(keepalive_due(stale, active, true));
    // Dead child → the reaper's job, not the sweep's.
    assert!(!keepalive_due(stale, active, false));
    // User idle past the keepalive horizon → stop; the idle reaper
    // takes the session anyway.
    let idle = Instant::now() - KEEPALIVE_IDLE_MAX - Duration::from_secs(1);
    assert!(!keepalive_due(stale, idle, true));
}

/// A stub whose stdin still writes and whose response channel is
/// caller-fed: enough wire for `send_prompt`/`await_prompt`.
fn wired_session() -> (LiveSession, std::sync::mpsc::Sender<Value>) {
    let mut s = stub_session();
    let (tx, rx) = std::sync::mpsc::channel::<Value>();
    s.rx = rx;
    (s, tx)
}

fn turn(model: &str, input: Vec<Value>) -> PreparedTurn {
    PreparedTurn {
        system: "sys".to_string(),
        content_line: String::new(),
        input,
        names: Vec::new(),
        native_model: model.to_string(),
        gray_session: String::new(),
        account: None,
    }
}

fn answered(text: &str) -> TurnResult {
    TurnResult {
        text: text.to_string(),
        thought: String::new(),
        calls: Vec::new(),
        usage: Usage::default(),
        stop: "completed".to_string(),
        unseen: false,
    }
}

/// The bug being fixed: a failed turn used to close the session, so the
/// host's identical whole-turn retry missed the pool and re-billed the
/// whole transcript. A failure whose prompt still settled (here, an RPC
/// error) leaves the absorb state untouched — the retry produces the
/// same delta on the warm session.
#[test]
fn a_settled_failure_keeps_the_retry_warm() {
    let model = "model-settled-fail";
    let history = vec![json!({"role": "user", "content": "hi"})];
    let first = turn(model, history.clone());
    let (mut s, tx) = wired_session();
    s.session_id = "s-settled".to_string();
    s.native_model = model.to_string();
    s.absorb(&first, &answered("answer"));

    // Turn 2 continues: history + our echo + a new user item.
    let mut input = history;
    input.push(json!({"role": "assistant", "content": "answer"}));
    input.push(json!({"role": "user", "content": "more"}));
    let next = turn(model, input.clone());
    let delta = continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &next.input)
        .expect("continuation");

    // The turn's prompt goes out, then settles with an RPC error.
    let id = s.send_prompt(&delta, &[], &[]).expect("send_prompt");
    assert!(s.prompt_in_flight);
    tx.send(json!({"jsonrpc": "2.0", "id": id, "error": {"message": "boom"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let e = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .err()
        .unwrap();
    assert!(e.contains("boom"), "{e}");
    assert!(
        !s.prompt_in_flight,
        "a settled failure leaves the session poolable"
    );

    // Pooled, the host's identical retry takes the same session and the
    // same delta — no transcript re-bill.
    give_back(s);
    let (claimed, reason) = take(&next, &never);
    let (mut s, retry_delta) = claimed.unwrap_or_else(|| panic!("retry missed the pool: {reason}"));
    assert_eq!(s.session_id, "s-settled");
    assert_eq!(retry_delta, delta);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// The closing half of the fix: a prompt that never settles (dead wire,
/// a cancel upstream ignores) leaves the session untrusted — it must be
/// closed, not pooled.
#[test]
fn an_unsettled_failure_cannot_be_pooled() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    assert!(s.prompt_in_flight);
    // The wire dies mid-prompt: nothing settles, the prompt may still be
    // live upstream.
    drop(tx);
    let never = AtomicBool::new(false);
    let e = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .err()
        .unwrap();
    assert_eq!(e, "native stdout closed");
    assert!(
        s.prompt_in_flight,
        "an unsettled prompt keeps the session untrusted"
    );
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// The settle grace after a failed wait: the prompt's reply lands inside
/// the cancel window — a known session state, and (client still present)
/// a deliverable reply.
#[test]
fn a_prompt_settling_in_the_grace_window_is_still_delivered() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let _ = tx.send(json!({"jsonrpc": "2.0", "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "late answer"}}}}));
        let _ = tx.send(json!({"jsonrpc": "2.0", "id": id,
            "result": {"stopReason": "end_turn"}}));
    });
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_millis(150), &never)
        .expect("late-settling reply must still return");
    assert!(!r.unseen, "the relay client never went away");
    assert!(!s.prompt_in_flight);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// A refusal is a settled prompt: the session state is known, so the
/// failure leaves the session poolable for the host's retry.
#[test]
fn a_refusal_leaves_the_session_poolable() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "refusal"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let e = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .err()
        .unwrap();
    assert!(e.contains("refused"), "{e}");
    assert!(!s.prompt_in_flight);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// Fresh-path failure bookkeeping: `absorb_failed` records the failed
/// input with no reply echo, so a later turn of THIS conversation
/// continues warm — while the identical retry respawns (nothing extends
/// the absorbed input) and no other conversation can claim the child.
#[test]
fn absorb_failed_marks_input_without_leaking_across_conversations() {
    let (mut s, _tx) = wired_session();
    let input = vec![json!({"role": "user", "content": "hi"})];
    s.absorb_failed(&turn("model", input.clone()));
    let cont = |input: &[Value]| continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, input);
    // The identical host retry cannot extend — it respawns fresh.
    assert_eq!(cont(&input), Err("prefix"));
    // An unrelated conversation can't claim a child already holding this
    // transcript either.
    let other = vec![json!({"role": "user", "content": "different convo"})];
    assert_eq!(cont(&other), Err("prefix"));
    // A later turn of the same conversation continues with only its tail.
    let mut next = input;
    next.push(json!({"role": "user", "content": "more"}));
    assert_eq!(cont(&next), Ok("[User]\nmore".to_string()));
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// The silent usage-limit rejection: a real prompt settles `end_turn`
/// with zero stream content — no text, no thought, no tool call. It must
/// surface as a failure (the host can't tell an empty "success" from a
/// dropped prompt), and the settled prompt leaves the session poolable.
#[test]
fn an_empty_end_turn_on_a_real_prompt_is_a_limit_failure() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let e = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .err()
        .unwrap();
    assert!(e.contains("empty end_turn"), "{e}");
    assert!(e.contains("usage limit"), "{e}");
    assert!(is_usage_limit(&e), "{e}");
    assert!(
        !s.prompt_in_flight,
        "the prompt settled — the session stays poolable"
    );
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// Keepalives ask for a near-empty reply by design: an empty `end_turn`
/// answers one fine and must never be flagged.
#[test]
fn a_keepalive_may_answer_empty() {
    let (mut s, tx) = wired_session();
    let id = s
        .send_prompt(KEEPALIVE_TEXT, &[], &[])
        .expect("send_prompt");
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .expect("an empty keepalive reply is fine");
    assert!(r.text.is_empty());
    assert_eq!(r.stop, "completed");
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// A turn whose `session/update` stream carried content is a real answer,
/// however short: not a silent drop.
#[test]
fn a_streamed_answer_is_not_flagged_empty() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    tx.send(json!({"jsonrpc": "2.0", "method": "session/update",
        "params": {"update": {"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "ok"}}}}))
        .unwrap();
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .expect("a streamed answer is a normal completion");
    assert_eq!(r.text, "ok");
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// Two message segments separated by a non-message update keep their
/// paragraph break — upstream emits each segment as its own chunk series,
/// and joining them raw fuses the boundary ("context.Now").
#[test]
fn message_segments_around_another_update_keep_their_break() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    for update in [
        json!({"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "first segment."}}),
        json!({"sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "thinking"}}),
        json!({"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Second segment."}}),
    ] {
        tx.send(json!({"jsonrpc": "2.0", "method": "session/update",
            "params": {"update": update}}))
            .unwrap();
    }
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .expect("completion");
    assert_eq!(r.text, "first segment.\n\nSecond segment.", "{:?}", r.text);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// Back-to-back message chunks in one segment still concatenate raw —
/// delta streaming is untouched.
#[test]
fn adjacent_message_chunks_concatenate_raw() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    for update in [
        json!({"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "one "}}),
        json!({"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "two."}}),
    ] {
        tx.send(json!({"jsonrpc": "2.0", "method": "session/update",
            "params": {"update": update}}))
            .unwrap();
    }
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .expect("completion");
    assert_eq!(r.text, "one two.", "{:?}", r.text);
    let _ = s.child.kill();
    let _ = s.child.wait();
}

/// Thought chunks are stream content too: a turn that reasoned but
/// answered with no text still isn't a silent drop.
#[test]
fn a_thought_only_turn_is_not_flagged_empty() {
    let (mut s, tx) = wired_session();
    let id = s.send_prompt("[User]\nhi", &[], &[]).expect("send_prompt");
    tx.send(json!({"jsonrpc": "2.0", "method": "session/update",
        "params": {"update": {"sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "hmm"}}}}))
        .unwrap();
    tx.send(json!({"jsonrpc": "2.0", "id": id,
        "result": {"stopReason": "end_turn"}}))
        .unwrap();
    let never = AtomicBool::new(false);
    let r = s
        .await_prompt(id, Instant::now() + Duration::from_secs(30), &never)
        .expect("streamed thought is content");
    assert_eq!(r.thought, "hmm");
    let _ = s.child.kill();
    let _ = s.child.wait();
}

fn record(session_id: &str, model: &str, owner_pid: u32, last_prompt_unix: u64) -> Persisted {
    Persisted {
        session_id: session_id.to_string(),
        native_model: model.to_string(),
        system: "sys".to_string(),
        absorbed: vec![json!({"role": "user", "content": "hi"})],
        reply_call_ids: Vec::new(),
        reply_text: "answer".to_string(),
        last_used_unix: last_prompt_unix,
        last_prompt_unix,
        owner_pid,
        gray_session: String::new(),
        deny_search: Some(false),
        account: String::new(),
        exact: true,
    }
}

fn continued() -> Vec<Value> {
    vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "answer"}),
        json!({"role": "user", "content": "more"}),
    ]
}

#[test]
fn reload_candidates_take_only_orphaned_fresh_matching_continuations() {
    let now = 10_000;
    let max_age = Duration::from_secs(30 * 60);
    let dead = |_: u32| false;
    let next = turn("m", continued());

    // The plain case: dead owner, young, same model/system, strict
    // continuation -> one candidate carrying the delta.
    let got = reload_candidates(
        vec![record("a", "m", 1, now - 60)],
        &next,
        now,
        max_age,
        &dead,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0.session_id, "a");
    assert!(
        got[0].1.contains("more"),
        "delta is the new tail: {}",
        got[0].1
    );

    // A live owner still holds the session: never load out from under it.
    let alive = |pid: u32| pid == 1;
    assert!(
        reload_candidates(
            vec![record("a", "m", 1, now - 60)],
            &next,
            now,
            max_age,
            &alive
        )
        .is_empty()
    );

    // Past max_age the cache is cold: a load buys nothing over fresh.
    let old = record("a", "m", 1, now - max_age.as_secs() - 1);
    assert!(reload_candidates(vec![old], &next, now, max_age, &dead).is_empty());
    let edge = record("a", "m", 1, now - max_age.as_secs());
    assert_eq!(
        reload_candidates(vec![edge], &next, now, max_age, &dead).len(),
        1
    );

    // Other model or other system text: not this conversation's child.
    assert!(
        reload_candidates(
            vec![record("a", "other", 1, now - 60)],
            &next,
            now,
            max_age,
            &dead
        )
        .is_empty()
    );
    let mut sys = record("a", "m", 1, now - 60);
    sys.system = "other sys".to_string();
    assert!(reload_candidates(vec![sys], &next, now, max_age, &dead).is_empty());

    // Not a strict continuation (the echo differs): skipped.
    let mut echo = record("a", "m", 1, now - 60);
    echo.reply_text = "something else".to_string();
    assert!(reload_candidates(vec![echo], &next, now, max_age, &dead).is_empty());

    // Several qualify: newest upstream contact first.
    let got = reload_candidates(
        vec![
            record("older", "m", 1, now - 300),
            record("newer", "m", 2, now - 30),
        ],
        &next,
        now,
        max_age,
        &dead,
    );
    assert_eq!(
        got.iter()
            .map(|(p, _)| p.session_id.as_str())
            .collect::<Vec<_>>(),
        ["newer", "older"]
    );
}

#[test]
fn a_same_conversation_record_is_claimable_under_a_live_owner() {
    let now = 10_000;
    let max_age = Duration::from_secs(30 * 60);
    let alive = |pid: u32| pid == 1;
    let mut next = turn("m", continued());
    next.gray_session = "gray-1".to_string();

    // Same gray session whose owner pid still looks alive (a dying
    // sidecar's handoff window, or a recycled pid): the record is the
    // rightful heir's to claim.
    let mut same = record("a", "m", 1, now - 60);
    same.gray_session = "gray-1".to_string();
    let got = reload_candidates(vec![same], &next, now, max_age, &alive);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0.session_id, "a");

    // A different live-owned conversation still refuses.
    let mut other = record("b", "m", 1, now - 60);
    other.gray_session = "gray-2".to_string();
    assert!(reload_candidates(vec![other], &next, now, max_age, &alive).is_empty());

    // A record written before the field existed can't be claimed past a
    // live owner, and an unidentified turn can't claim a live-owned one.
    let legacy = record("c", "m", 1, now - 60);
    assert!(reload_candidates(vec![legacy], &next, now, max_age, &alive).is_empty());
    let mut anon_record = record("d", "m", 1, now - 60);
    anon_record.gray_session = "gray-1".to_string();
    let anon_turn = turn("m", continued());
    assert!(reload_candidates(vec![anon_record], &anon_turn, now, max_age, &alive).is_empty());
}

#[test]
fn persisted_record_round_trips_and_mirrors_live_state() {
    let mut s = stub_session();
    s.session_id = "s-persist".to_string();
    s.native_model = "m".to_string();
    // Never prompted: nothing upstream worth a record.
    assert!(s.persisted().is_none());
    s.absorb(
        &turn("m", vec![json!({"role": "user", "content": "hi"})]),
        &answered("answer"),
    );
    let p = s.persisted().expect("a prompted session persists");
    assert_eq!(p.session_id, "s-persist");
    assert_eq!(p.native_model, "m");
    assert_eq!(p.system, "sys");
    assert_eq!(p.absorbed, vec![json!({"role": "user", "content": "hi"})]);
    assert_eq!(p.reply_text, "answer");
    assert_eq!(p.owner_pid, std::process::id());
    assert!(unix_now() - p.last_prompt_unix <= 1);
    let back: Persisted = serde_json::from_slice(&serde_json::to_vec(&p).unwrap()).unwrap();
    assert_eq!(back, p);
    // Our own pid counts as a live owner: a record we wrote for a session
    // currently out of our pool is never reloaded into a second child.
    assert!(pid_alive(std::process::id()));
    // What the record says this conversation continues with is exactly
    // what the live pool would have said.
    let next = turn("m", continued());
    let got = reload_candidates(
        vec![back],
        &next,
        unix_now(),
        Duration::from_secs(60),
        &|_| false,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].1,
        continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &next.input).unwrap()
    );
    s.close();
}

#[test]
fn usage_limit_errors_are_the_ones_not_worth_retrying() {
    assert!(is_usage_limit(
        "Devin usage limit reached: Reached free model rate limit. Please switch"
    ));
    assert!(is_usage_limit("HTTP 429 Too Many Requests"));
    assert!(!is_usage_limit("prompt timed out after 60s"));
    assert!(!is_usage_limit("child exited"));
}

#[test]
fn limit_line_matches_only_the_refusal_signature() {
    for line in [
        "devin subscription usage limit reached (resets 18:00)",
        "POST /v1/turns -> 429 Too Many Requests",
        "daily quota exhausted, retrying in 240s",
        "upstream rate limit exceeded",
    ] {
        assert!(limit_line(line), "{line}");
    }
    for line in [
        "session/load ok",
        "prompt settled end_turn",
        "permission request id=9",
    ] {
        assert!(!limit_line(line), "{line}");
    }
}

#[test]
fn early_abort_error_still_maps_to_429() {
    // limit_error's output must satisfy is_usage_limit, else the relay's
    // 429 mapping misses it and the host retries the wedged turn.
    assert!(is_usage_limit("Devin usage limit reached"));
    assert!(is_usage_limit(
        "Devin usage limit reached: daily quota exhausted"
    ));
}

/// A reasoning-only reply (no text, no tool call) must not be replayed:
/// the host's identical retry of that empty completion has to reach a
/// real request, or it gets the same empty answer back forever.
#[test]
fn an_empty_stored_reply_is_not_replayed() {
    let model = "model-empty-replay";
    let input = vec![json!({"role": "user", "content": "go"})];
    let t = turn(model, input.clone());
    let mut s = stub_session();
    s.session_id = "s-empty-replay".to_string();
    s.native_model = model.to_string();
    let mut empty = answered("");
    empty.thought = "thinking only".to_string();
    s.absorb(&t, &empty);
    give_back(s);
    assert!(replay(&t).is_none());

    // The same request with a real stored answer still replays.
    let mut pool = locked_pool();
    let s = pool
        .live
        .iter_mut()
        .find(|s| s.session_id == "s-empty-replay")
        .expect("pooled");
    s.last_reply = Some(answered("real"));
    drop(pool);
    assert_eq!(replay(&t).map(|r| r.text).as_deref(), Some("real"));
    let mut pool = locked_pool();
    let i = pool
        .live
        .iter()
        .position(|s| s.session_id == "s-empty-replay")
        .expect("pooled");
    let s = pool.live.remove(i);
    drop(pool);
    s.close();
}
