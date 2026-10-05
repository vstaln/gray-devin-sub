use super::*;
use serde_json::json;

fn bash_tool() -> Value {
    json!({"type": "function", "name": "bash",
        "description": "run a shell command",
        "parameters": {"type": "object",
            "properties": {"command": {"type": "string"}}, "required": ["command"]}})
}

#[test]
fn prepare_accepts_short_form_user_message() {
    let body = json!({"input": [{"role": "user", "content": "hi"}]});
    let turn = prepare_turn(&body, "swe-2-medium").expect("short-form message must prepare");
    assert!(turn.content_line.contains("hi"), "{}", turn.content_line);
    assert_eq!(turn.native_model, "swe-2-medium");
}

#[test]
fn prepare_accepts_short_form_assistant_then_user() {
    let body = json!({"input": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}
    ]});
    let turn = prepare_turn(&body, "swe-2-medium").expect("replay must prepare");
    assert!(turn.content_line.contains("hello"));
    assert!(turn.content_line.contains("again"));
}

#[test]
fn prepare_rejects_assistant_prefill() {
    let body = json!({"input": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]});
    assert!(prepare_turn(&body, "swe-2-medium").is_err());
}

#[test]
fn prepare_collects_tool_names_into_contract() {
    let body = json!({"input": [{"role": "user", "content": "hi"}],
        "tools": [bash_tool()]});
    let turn = prepare_turn(&body, "swe-2-medium").unwrap();
    assert!(turn.system.contains("gray_calls"));
    assert!(turn.system.contains("bash"));
    assert_eq!(turn.names, vec!["bash".to_string()]);
}

#[test]
fn funnel_parse_valid_block_with_leading_text() {
    let text = "let me run that\n```gray_calls\n[{\"name\": \"bash\", \"arguments\": {\"command\": \"echo hi\"}}]\n```";
    let (before, calls) =
        parse_calls_block(text, &["bash".to_string()]).expect("valid block must parse");
    assert_eq!(before, "let me run that");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "bash");
    assert!(calls[0].1.contains("echo hi"));
}

#[test]
fn funnel_parse_missing_block_is_none() {
    assert!(parse_calls_block("just an answer", &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_bad_json_is_none() {
    let text = "```gray_calls\nnot json\n```";
    assert!(parse_calls_block(text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_unknown_tool_is_none() {
    let text = "```gray_calls\n[{\"name\": \"hack\", \"arguments\": {}}]\n```";
    assert!(parse_calls_block(text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_non_object_args_is_none() {
    let text = "```gray_calls\n[{\"name\": \"bash\", \"arguments\": [\"x\"]}]\n```";
    assert!(parse_calls_block(text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_markup_terminated_block() {
    // SWE models end the call list with native pipe-delimited tool markup
    // instead of the ``` fence (and may hallucinate transcript turns after
    // it). The array still parses; everything from the fence on is dropped.
    let markup = concat!(
        "<",
        "|close",
        "|>argument<",
        "|sep",
        "|><",
        "|close",
        "|>call<",
        "|sep",
        "|><",
        "|close",
        "|>tools<",
        "|sep",
        "|>"
    );
    let text = format!(
        "checking the file\n```gray_calls\n[{{\"name\": \"bash\", \"arguments\": {{\"command\": \"sed -n '1,5p' x\"}}}}]{markup}\n\n[User]\nContinue.\n\n[Assistant]\nRetrying."
    );
    let (before, calls) = parse_calls_block(&text, &["bash".to_string()])
        .expect("markup-terminated block must parse");
    assert_eq!(before, "checking the file");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "bash");
    assert!(calls[0].1.contains("sed -n"));
}

#[test]
fn funnel_parse_unclosed_block_at_eof() {
    let text = "reading it now\n```gray_calls\n[{\"name\": \"bash\", \"arguments\": {\"command\": \"cat f\"}}]";
    let (before, calls) =
        parse_calls_block(text, &["bash".to_string()]).expect("unclosed block must parse");
    assert_eq!(before, "reading it now");
    assert_eq!(calls[0].0, "bash");
}

#[test]
fn funnel_parse_junk_after_array_inside_fence() {
    let text = "```gray_calls\n[{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}] stray words\n```";
    let (_, calls) =
        parse_calls_block(text, &["bash".to_string()]).expect("trailing junk in block is dropped");
    assert_eq!(calls[0].0, "bash");
}

#[test]
fn funnel_parse_unclosed_garbage_is_none() {
    let text = "thinking out loud\n```gray_calls\n[{not json";
    assert!(parse_calls_block(text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_uses_last_block() {
    let text = "```gray_calls\n[{\"name\": \"bash\", \"arguments\": {\"command\": \"first\"}}]\n```\n\
                wait\n```gray_calls\n[{\"name\": \"bash\", \"arguments\": {\"command\": \"last\"}}]\n```";
    let (before, calls) = parse_calls_block(text, &["bash".to_string()]).unwrap();
    assert!(calls[0].1.contains("last"));
    assert!(before.contains("first"));
}

#[test]
fn redirect_execute_maps_command() {
    let u = json!({"kind": "execute", "rawInput": {"command": "echo hi"}});
    assert_eq!(
        redirect_call(&u, true).unwrap(),
        "{\"command\":\"echo hi\"}"
    );
}

#[test]
fn redirect_read_quotes_single_quote_path() {
    let u = json!({"kind": "read", "rawInput": {"file_path": "/tmp/it's here.txt"}});
    let args = redirect_call(&u, true).unwrap();
    assert_eq!(args, "{\"command\":\"cat -- '/tmp/it'\\\\''s here.txt'\"}");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["command"], "cat -- '/tmp/it'\\''s here.txt'");
}

#[test]
fn redirect_unknown_kind_ignored() {
    let u = json!({"kind": "edit", "rawInput": {"file_path": "/x"}});
    assert!(redirect_call(&u, true).is_none());
    assert!(redirect_call(&u, false).is_none());
}

#[test]
fn fold_emits_complete_function_call_items() {
    let r = TurnResult {
        text: "done".to_string(),
        thought: String::new(),
        calls: vec![(
            "call_1_1".to_string(),
            "bash".to_string(),
            "{\"command\":\"echo hi\"}".to_string(),
        )],
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
            cached_tokens: 3,
            cache_write_tokens: 4,
        },
        stop: "tool_use".to_string(),
        unseen: false,
    };
    let sse = String::from_utf8(fold_result(&r, &["bash".to_string()]).unwrap()).unwrap();
    assert!(sse.contains("response.output_item.added"));
    assert!(sse.contains("\"cache_creation_tokens\":4"));
    assert!(sse.contains("response.function_call_arguments.done"));
    assert!(sse.contains("response.output_item.done"));
    assert!(sse.contains("response.completed"));
    assert!(sse.contains("\"call_id\":\"call_1_1\""));
}

#[test]
fn fold_rejects_tool_outside_inventory() {
    let r = TurnResult {
        text: String::new(),
        thought: String::new(),
        calls: vec![("call_1_1".into(), "hack".into(), "{}".into())],
        usage: Usage::default(),
        stop: "tool_use".to_string(),
        unseen: false,
    };
    assert!(fold_result(&r, &["bash".to_string()]).is_err());
}

#[test]
fn shell_quote_handles_plain_path() {
    assert_eq!(shell_quote("/etc/hostname"), "'/etc/hostname'");
}

#[test]
fn fold_marks_max_tokens_incomplete() {
    let r = TurnResult {
        text: "cut off".to_string(),
        thought: String::new(),
        calls: Vec::new(),
        usage: Usage::default(),
        stop: "incomplete:max_output_tokens".to_string(),
        unseen: false,
    };
    let sse = String::from_utf8(fold_result(&r, &[]).unwrap()).unwrap();
    assert!(sse.contains("\"status\":\"incomplete\""));
    assert!(sse.contains("\"reason\":\"max_output_tokens\""));
}

fn absorbed() -> Vec<Value> {
    vec![
        json!({"role": "user", "content": "run ls"}),
        json!({"role": "assistant", "content": "sure"}),
    ]
}

/// The host's replay of an answer of `text` + one `call_1` bash call.
fn echo(text: &str, call_id: &str) -> Vec<Value> {
    vec![
        json!({"role": "assistant", "content": text}),
        json!({"type": "function_call", "call_id": call_id,
            "name": "bash", "arguments": "{\"command\":\"ls\"}"}),
    ]
}

#[test]
fn continuation_match_returns_delta() {
    let mut input = absorbed();
    input.extend(echo("on it", "call_1"));
    input.push(json!({"type": "function_call_output", "call_id": "call_1",
        "output": "file.txt"}));
    input.push(json!({"role": "user", "content": "now grep it"}));
    let delta = continuation(&absorbed(), &["call_1".to_string()], "on it", &input)
        .expect("strict continuation must match");
    assert!(delta.contains("[Tool result id=call_1]"), "{delta}");
    assert!(delta.contains("[User]\nnow grep it"), "{delta}");
    assert!(!delta.contains("run ls"), "delta must not repeat history");
}

#[test]
fn continuation_prefix_changed() {
    let mut input = absorbed();
    input[0]["content"] = json!("run pwd");
    input.extend(echo("on it", "call_1"));
    input.push(json!({"type": "function_call_output", "call_id": "call_1",
        "output": "x"}));
    assert_eq!(
        continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap_err(),
        "prefix"
    );
}

#[test]
fn continuation_echo_call_id_mismatch() {
    let mut input = absorbed();
    input.extend(echo("on it", "call_OTHER"));
    input.push(
        json!({"type": "function_call_output", "call_id": "call_OTHER",
        "output": "x"}),
    );
    assert_eq!(
        continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap_err(),
        "echo"
    );
}

#[test]
fn continuation_echo_text_mismatch() {
    let mut input = absorbed();
    input.extend(echo("different words", "call_1"));
    input.push(json!({"type": "function_call_output", "call_id": "call_1",
        "output": "x"}));
    assert_eq!(
        continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap_err(),
        "echo"
    );
}

#[test]
fn continuation_assistant_item_in_new_tail() {
    // A second assistant message after the tool result means the history
    // mid-edited a turn — not a strict continuation.
    let mut input = absorbed();
    input.extend(echo("on it", "call_1"));
    input.push(json!({"type": "function_call_output", "call_id": "call_1",
        "output": "x"}));
    input.push(json!({"role": "assistant", "content": "sneaky"}));
    input.push(json!({"role": "user", "content": "go on"}));
    assert_eq!(
        continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap_err(),
        "empty_delta"
    );
}

#[test]
fn continuation_empty_tail() {
    // Echo of our answer and nothing else: nothing new to ask.
    let mut input = absorbed();
    input.extend(echo("on it", "call_1"));
    assert_eq!(
        continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap_err(),
        "empty_delta"
    );
}

#[test]
fn continuation_input_not_longer() {
    assert_eq!(
        continuation(&absorbed(), &[], "", &absorbed()).unwrap_err(),
        "prefix"
    );
    assert_eq!(
        continuation(&absorbed(), &[], "", &absorbed()[..1]).unwrap_err(),
        "prefix"
    );
}

#[test]
fn continuation_text_only_reply_echo() {
    // A calls-free reply is echoed as the assistant message alone.
    let mut input = absorbed();
    input.push(json!({"role": "assistant", "content": "done"}));
    input.push(json!({"role": "user", "content": "next"}));
    let delta = continuation(&absorbed(), &[], "done", &input).unwrap();
    assert_eq!(delta, "[User]\nnext");
}

#[test]
fn continuation_unseen_reply_echoes_nothing() {
    // A turn that settled after the relay client was gone records an
    // empty reply (LiveSession::absorb, r.unseen): gray's history carries
    // no assistant items for it, so the echo zone is empty and the new
    // tail alone is the delta.
    let mut input = absorbed();
    input.push(json!({"role": "user", "content": "retry that"}));
    let delta = continuation(&absorbed(), &[], "", &input).unwrap();
    assert_eq!(delta, "[User]\nretry that");
}

#[test]
fn usage_from_flat_cognition_meta() {
    let u = usage_from(&json!({"sessionUpdate": "usage_update",
        "_meta": {"cognition.ai/inputTokens": 36304,
            "cognition.ai/outputTokens": 27,
            "cognition.ai/cachedReadTokens": 36224}}))
    .expect("flat cognition.ai keys must parse");
    assert_eq!(u.input_tokens, 36304);
    assert_eq!(u.output_tokens, 27);
    assert_eq!(u.cached_tokens, 36224);
}

#[test]
fn usage_from_reads_cache_writes() {
    // Real first-turn update: no cached read, only a cache write.
    let u = usage_from(&json!({"sessionUpdate": "usage_update",
        "_meta": {"cognition.ai/inputTokens": 5620,
            "cognition.ai/outputTokens": 25,
            "cognition.ai/cachedWriteTokens": 5617}}))
    .unwrap();
    assert_eq!(u.cached_tokens, 0);
    assert_eq!(u.cache_write_tokens, 5617);
    let u = usage_from(&json!({"cachedWriteTokens": 7})).expect("write-only usage is usage");
    assert_eq!(u.cache_write_tokens, 7);
}

#[test]
fn usage_from_nested_cognition_meta() {
    let u = usage_from(&json!({"_meta": {"cognition.ai": {
        "inputTokens": 10, "outputTokens": 2, "cachedReadTokens": 8}}}))
    .unwrap();
    assert_eq!(u.input_tokens, 10);
    assert_eq!(u.cached_tokens, 8);
}

#[test]
fn usage_from_plain_keys() {
    let u = usage_from(&json!({"totalTokens": 36331, "inputTokens": 36304,
        "outputTokens": 27, "cachedReadTokens": 36224}))
    .unwrap();
    assert_eq!(u.input_tokens, 36304);
    assert_eq!(u.cached_tokens, 36224);
}

#[test]
fn usage_from_absent_or_zero_is_none() {
    assert!(usage_from(&json!({"sessionUpdate": "usage_update"})).is_none());
    assert!(usage_from(&json!({"inputTokens": 0, "outputTokens": 0})).is_none());
}
