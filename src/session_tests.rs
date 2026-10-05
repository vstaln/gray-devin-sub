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
