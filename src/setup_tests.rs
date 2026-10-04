use super::*;

#[test]
fn auth_status_logged_in() {
    assert_eq!(
        auth_status_verdict(true, "Logged in (via Devin) as someone@example.com"),
        LoginState::LoggedIn
    );
}

#[test]
fn auth_status_not_logged_in() {
    assert_eq!(
        auth_status_verdict(true, "Not logged in. Run `devin auth login`."),
        LoginState::LoggedOut
    );
    assert_eq!(
        auth_status_verdict(true, "not authenticated"),
        LoginState::LoggedOut
    );
    assert_eq!(
        auth_status_verdict(false, "Logged in (via Devin)"),
        LoginState::LoggedOut
    );
}

#[test]
fn child_env_strips_devin_overrides() {
    let _env = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for key in [
        "DEVIN_MODEL",
        "DEVIN_PERMISSION_MODE",
        "DEVIN_SANDBOX",
        "DEVIN_REFUSAL_FALLBACK",
    ] {
        unsafe { std::env::set_var(key, "x") };
    }
    let env = child_env();
    for key in [
        "DEVIN_MODEL",
        "DEVIN_PERMISSION_MODE",
        "DEVIN_SANDBOX",
        "DEVIN_REFUSAL_FALLBACK",
    ] {
        unsafe { std::env::remove_var(key) };
        assert!(!env.iter().any(|(k, _)| k == key), "{key} leaked");
    }
}
