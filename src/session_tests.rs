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
        prompt_in_flight: false,
        text: String::new(),
        text_boundary: false,
        thought: String::new(),
        redirected: Vec::new(),
        usage: Usage::default(),
        names: Vec::new(),
        keepalive_prompt: false,
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
fn a_turn_waits_for_its_session_to_return_from_a_keepalive() {
    // Unique model: the pool is process-wide and tests run in parallel.
    let model = "model-keepalive-wait";
    let turn = |model: &str, input: Vec<Value>| PreparedTurn {
        system: "sys".to_string(),
        content_line: String::new(),
        input,
        names: Vec::new(),
        native_model: model.to_string(),
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
    assert!(e.contains("usage-limit"), "{e}");
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
