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
        },
        stop: "tool_use".to_string(),
    };
    let sse = String::from_utf8(fold_result(&r, &["bash".to_string()]).unwrap()).unwrap();
    assert!(sse.contains("response.output_item.added"));
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
    };
    let sse = String::from_utf8(fold_result(&r, &[]).unwrap()).unwrap();
    assert!(sse.contains("\"status\":\"incomplete\""));
    assert!(sse.contains("\"reason\":\"max_output_tokens\""));
}
