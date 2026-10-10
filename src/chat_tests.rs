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
fn funnel_parse_bad_json_passes_raw_block() {
    // A malformed block still goes to the host, which answers "arguments
    // must be a valid JSON object" instead of the turn ending on raw JSON.
    let text =
        "fixing it\n```gray_calls\n[{\"name\":\"bash\",\"arguments\":{\"command\":\"a\"b\"}}]\n```";
    let (before, calls) = parse_calls_block(text, &["bash".to_string()]).unwrap();
    assert_eq!(before, "fixing it");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "bash");
    assert!(serde_json::from_str::<serde_json::Value>(&calls[0].1).is_err());
}

#[test]
fn funnel_parse_unknown_tool_folds_native() {
    // Default surface is bash-only; models still reach for `edit`. The
    // unadvertised name folds to `native__edit`, so the host's
    // "does not exist" error — never an execution — answers it.
    let text = "Adding it to BUSY.\n\n```gray_calls\n[{\"name\":\"edit\",\"arguments\":{\"file_path\":\"/tmp/d.py\",\"old_string\":\"a\",\"new_string\":\"b\"}}]\n```";
    let (before, calls) = parse_calls_block(text, &["bash".to_string()]).unwrap();
    assert_eq!(before, "Adding it to BUSY.");
    assert_eq!(calls[0].0, "native__edit");
    assert!(calls[0].1.contains("old_string"));
}

#[test]
fn funnel_parse_non_object_args_pass_through() {
    let text = "```gray_calls\n[{\"name\": \"bash\", \"arguments\": [\"x\"]}]\n```";
    let (_, calls) = parse_calls_block(text, &["bash".to_string()]).unwrap();
    assert_eq!(calls[0], ("bash".to_string(), "[\"x\"]".to_string()));
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
fn funnel_parse_unclosed_garbage_is_one_raw_call() {
    let text = "thinking out loud\n```gray_calls\n[{not json";
    let (_, calls) = parse_calls_block(text, &["bash".to_string()]).unwrap();
    assert_eq!(calls, vec![("bash".to_string(), "[{not json".to_string())]);
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
        redirect_call(&u, &["bash".to_string()]).unwrap(),
        ("bash".to_string(), "{\"command\":\"echo hi\"}".to_string())
    );
}

#[test]
fn redirect_read_quotes_single_quote_path() {
    let u = json!({"kind": "read", "rawInput": {"file_path": "/tmp/it's here.txt"}});
    let (name, args) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "bash");
    assert_eq!(args, "{\"command\":\"cat -- '/tmp/it'\\\\''s here.txt'\"}");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["command"], "cat -- '/tmp/it'\\''s here.txt'");
}

#[test]
fn redirect_unknown_kind_ignored() {
    let u = json!({"kind": "fetch", "rawInput": {"url": "https://x"}});
    assert!(redirect_call(&u, &[]).is_none());
    // edit without old_string can't map either
    let u = json!({"kind": "edit", "rawInput": {"file_path": "/x"}});
    assert!(redirect_call(&u, &["bash".to_string()]).is_none());
}

#[test]
fn redirect_search_maps_to_host_web_search() {
    let u = json!({"kind": "search", "rawInput": {"query": "x y"}});
    let names = vec!["bash".to_string(), "web_search".to_string()];
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "web_search");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["query"], "x y");
    // without the host tool it folds to a self-explaining native__ name
    let (name, _) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "native__web_search_not_enabled_for_devin");
}

#[test]
fn redirect_edit_maps_to_python_replace() {
    let u = json!({"kind": "edit", "rawInput": {"file_path": "/tmp/f.txt",
        "old_string": "a'b", "new_string": "c"}});
    let (name, args) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "bash");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    let cmd = parsed["command"].as_str().unwrap();
    assert!(cmd.starts_with("python3 -c "), "{cmd}");
    // JSON payload survives shell quoting verbatim
    let payload = cmd.rsplit(' ').next().unwrap();
    assert!(payload.starts_with('\''), "{payload}");
    let inner = payload[1..payload.len() - 1].replace("'\\''", "'");
    let a: Value = serde_json::from_str(&inner).unwrap();
    assert_eq!(a["old_string"], "a'b");
    assert_eq!(a["file_path"], "/tmp/f.txt");
}

#[test]
fn redirect_write_maps_to_python_write() {
    let u = json!({"kind": "write", "rawInput": {"file_path": "/tmp/f.txt", "content": "hi\n"}});
    let (name, args) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "bash");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert!(
        parsed["command"]
            .as_str()
            .unwrap()
            .starts_with("python3 -c ")
    );
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
    let sse = String::from_utf8(fold_result(&r).unwrap()).unwrap();
    assert!(sse.contains("response.output_item.added"));
    assert!(sse.contains("\"cache_creation_tokens\":4"));
    assert!(sse.contains("response.function_call_arguments.done"));
    assert!(sse.contains("response.output_item.done"));
    assert!(sse.contains("response.completed"));
    assert!(sse.contains("\"call_id\":\"call_1_1\""));
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
    let sse = String::from_utf8(fold_result(&r).unwrap()).unwrap();
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

#[test]
fn continuation_undelivered_reply_continues_with_note() {
    // The session recorded a reply gray never got (interrupted before
    // delivery): the echo zone is empty, yet the session is reused.
    let mut input = absorbed();
    input.push(json!({"role": "user", "content": "stop, do this instead"}));
    let delta = continuation(&absorbed(), &["call_1".to_string()], "on it", &input).unwrap();
    assert_eq!(
        delta,
        format!("{UNDELIVERED_NOTE}\n\n[User]\nstop, do this instead")
    );
}

#[test]
fn inline_images_become_acp_blocks() {
    let input = vec![json!({"role": "user", "content": [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K"},
        {"type": "input_image", "image_url": "https://example.com/x.png"},
    ]})];
    assert_eq!(
        images_of(&input),
        vec![json!({"type": "image", "mimeType": "image/png", "data": "iVBORw0K"})]
    );
    assert_eq!(
        render_items(&input),
        "[User]\nwhat is this[image attached][image omitted]"
    );
}

#[test]
fn funnel_parse_inline_mention_is_not_a_call() {
    // A fence quoted mid-paragraph is prose about the contract, not a
    // call: earlier it hijacked the parse and fed the trailing paragraph
    // to bash as unparseable arguments.
    let f = format!("{0}{0}{0}gray_calls", '`');
    let text = format!("(the {f} contract), not as native tools — use the fenced shape");
    assert!(parse_calls_block(&text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_fenced_prose_body_is_not_a_call() {
    let f = format!("{0}{0}{0}gray_calls", '`');
    let text = format!("explaining the shape:\n{f}\nname goes here, arguments there\n```");
    assert!(parse_calls_block(&text, &["bash".to_string()]).is_none());
}

#[test]
fn funnel_parse_real_block_beats_trailing_mention() {
    let f = format!("{0}{0}{0}gray_calls", '`');
    let text = format!(
        "{f}\n[{{\"name\":\"bash\",\"arguments\":{{\"command\":\"ls\"}}}}]\n```\n\n(above: a {f} block)"
    );
    let (_, calls) = parse_calls_block(&text, &["bash".to_string()]).unwrap();
    assert_eq!(calls[0].0, "bash");
    assert!(calls[0].1.contains("ls"));
}

#[test]
fn funnel_parse_empty_array_is_a_plain_answer() {
    let f = format!("{0}{0}{0}gray_calls", '`');
    let text = format!("nothing to run\n{f}\n[]\n```");
    let (before, calls) = parse_calls_block(&text, &["bash".to_string()]).unwrap();
    assert_eq!(before, "nothing to run");
    assert!(calls.is_empty());
}

#[test]
fn blocked_call_echoes_native_name_prefixed() {
    let u = json!({"title": "mcp__rea__binary_session", "kind": "other"});
    assert_eq!(blocked_call(&u).0, "native__mcp__rea__binary_session");
    let u = json!({"kind": "edit"});
    assert_eq!(blocked_call(&u).0, "native__edit");
    assert_eq!(blocked_call(&json!({})).0, "native__tool_call");
}

// Real `devin acp` shapes (probed): search AND fetch both arrive as
// kind "fetch", told apart by rawInput / _meta inferenceToolName.
#[test]
fn redirect_real_devin_search_shape() {
    let u = json!({"sessionUpdate": "tool_call", "title": "Searched web for Welch Labs transformers",
        "kind": "fetch", "rawInput": {"query": "Welch Labs transformers"},
        "_meta": {"cognition.ai/inferenceToolName": "web_search"}});
    let names = vec![
        "bash".to_string(),
        "web_search".to_string(),
        "web_fetch".to_string(),
    ];
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "web_search");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["query"], "Welch Labs transformers");
    let (name, _) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "native__web_search_not_enabled_for_devin");
}

#[test]
fn redirect_real_devin_fetch_shape() {
    let u = json!({"sessionUpdate": "tool_call", "title": "Fetched https://example.com",
        "kind": "fetch", "rawInput": {"url": "https://example.com"},
        "_meta": {"cognition.ai/inferenceToolName": "webfetch"}});
    let names = vec!["bash".to_string(), "web_fetch".to_string()];
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "web_fetch");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["url"], "https://example.com");
    // bash-only policy: curl through bash instead of a native__ dead end
    let (name, args) = redirect_call(&u, &["bash".to_string()]).unwrap();
    assert_eq!(name, "bash");
    let parsed: Value = serde_json::from_str(&args).unwrap();
    let cmd = parsed["command"].as_str().unwrap();
    assert!(cmd.starts_with("curl -fsSL"), "{cmd}");
    assert!(cmd.contains("-- 'https://example.com'"), "{cmd}");
}

#[test]
fn mcp_call_to_harness_tool_unwraps() {
    let names = vec!["bash".to_string(), "recall".to_string()];
    let u = json!({"kind": "other", "title": "Calling bash from harness",
        "rawInput": {"server_name": "harness", "tool_name": "bash",
            "arguments": {"command": "echo hi"}}});
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "bash");
    assert_eq!(
        serde_json::from_str::<Value>(&args).unwrap()["command"],
        "echo hi"
    );

    // Stringified arguments, a different "server", another allowed tool.
    let u = json!({"title": "Calling recall from gray",
        "rawInput": {"server_name": "gray", "tool_name": "recall",
            "arguments": "{\"query\":\"x\"}"}});
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "recall");
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap()["query"], "x");
}

#[test]
fn mcp_call_web_fetch_falls_back_to_curl_and_foreign_tools_stay_blocked() {
    let names = vec!["bash".to_string()];
    let u = json!({"title": "Calling web_fetch from harness",
        "rawInput": {"server_name": "harness", "tool_name": "web_fetch",
            "arguments": {"url": "https://example.com"}}});
    let (name, args) = redirect_call(&u, &names).unwrap();
    assert_eq!(name, "bash");
    assert!(args.contains("curl"), "{args}");

    let u = json!({"title": "Calling fetch_actor_details from apify",
        "rawInput": {"server_name": "apify", "tool_name": "fetch_actor_details",
            "arguments": {}}});
    assert!(redirect_call(&u, &names).is_none());
}
