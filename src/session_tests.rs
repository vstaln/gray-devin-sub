use super::*;
use serde_json::json;

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
    let mut child = Command::new("sleep")
        .arg("60")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("sleep");
    let stdin = child.stdin.take().expect("stdin");
    let (_tx, rx) = std::sync::mpsc::channel::<Value>();
    LiveSession {
        child,
        stdin,
        rx,
        stderr_tail: stderr(&[]),
        next_id: 0,
        session_id: "s-test".to_string(),
        native_model: "model".to_string(),
        system: "sys".to_string(),
        absorbed: Vec::new(),
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        last_used: Instant::now(),
        last_prompt_at: None,
        keepalive_turns: 0,
        text: String::new(),
        thought: String::new(),
        redirected: Vec::new(),
        usage: Usage::default(),
        names: Vec::new(),
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
    // User idle past the keepalive horizon → stop; the 15min reaper
    // takes the session anyway.
    let idle = Instant::now() - KEEPALIVE_IDLE_MAX - Duration::from_secs(1);
    assert!(!keepalive_due(stale, idle, true));
}
