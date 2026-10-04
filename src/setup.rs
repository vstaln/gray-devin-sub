//! Setup-time probes of the user's `devin` CLI: binary presence and login
//! state. `devin` owns its own auth (`devin auth login`, credentials inside
//! the CLI): this crate never reads, copies, or forwards any credential
//! material — it only probes whether a login exists and fails closed
//! otherwise.
//!
//! Login probe design: `devin auth status` exits 0 even when logged out,
//! so the verdict comes from its text — logged in iff exit 0 AND the
//! lowercased output contains "logged in" AND contains none of
//! "not logged in" / "not authenticated". (Verified real output when
//! logged in starts "Logged in (via Devin)".) Output is captured and
//! never echoed anywhere.

use std::path::PathBuf;

/// `devin` is missing (or not on PATH): install hint, never a spawn panic.
pub const INSTALL_HINT: &str = "`devin` not found on PATH. Install the Devin CLI, then run \
    `devin auth login` once to sign in. \
    Override the binary with GRAY_DEVIN_SUB_COMMAND=/path/to/devin.";
/// `devin` is present but has no usable login here.
pub const LOGIN_HINT: &str =
    "Devin CLI is installed but not logged in. Run `devin auth login`, then retry.";

fn env_override() -> Option<String> {
    std::env::var("GRAY_DEVIN_SUB_COMMAND")
        .ok()
        .filter(|v| !v.is_empty())
}

/// Resolve the `devin` binary: explicit override, then PATH.
pub fn resolve_command() -> Option<String> {
    if let Some(v) = env_override() {
        return Some(v);
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["devin", "devin.exe", "devin.cmd"] {
            let p: PathBuf = dir.join(name);
            if p.is_file() {
                return Some(p.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// Env keys that would override the turn's model, permission posture or
/// sandbox: strip them from the child's environment so the sidecar's own
/// flags always win.
const CONFLICTING_KEYS: &[&str] = &[
    "DEVIN_MODEL",
    "DEVIN_PERMISSION_MODE",
    "DEVIN_SANDBOX",
    "DEVIN_REFUSAL_FALLBACK",
];

/// Child env for every spawn: never inherit a conflicting value.
pub fn child_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(k, _)| !CONFLICTING_KEYS.contains(&k.as_str()))
        .collect()
}

/// Login state. `Unknown` degrades to an error, never to a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginState {
    LoggedIn,
    LoggedOut,
    Unknown,
}

/// Verdict from `devin auth status` output: the command exits 0 even when
/// logged out, so the text decides. Logged in iff the lowercased output
/// contains "logged in" and none of the negative phrases.
pub fn auth_status_verdict(exit_ok: bool, output: &str) -> LoginState {
    if !exit_ok {
        return LoginState::LoggedOut;
    }
    let text = output.to_lowercase();
    if text.contains("logged in")
        && !text.contains("not logged in")
        && !text.contains("not authenticated")
    {
        LoginState::LoggedIn
    } else {
        LoginState::LoggedOut
    }
}

/// Probe login state with `devin auth status` (8s cap; output captured and
/// never echoed — it may carry account details).
/// * exit 0 + "logged in" text → logged in.
/// * exit 0 without the phrase, or nonzero → logged out.
/// * spawn failure / timeout → `Unknown`.
pub fn probe_login() -> LoginState {
    let Some(binary) = resolve_command() else {
        return LoginState::Unknown;
    };
    let mut child = match std::process::Command::new(&binary)
        .args(["auth", "status"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .envs(child_env())
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return LoginState::Unknown,
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => break None,
            Err(_) => break None,
        }
    };
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        return LoginState::Unknown;
    };
    let mut buf = String::new();
    if let Some(mut out) = child.stdout.take() {
        use std::io::Read;
        let _ = out.read_to_string(&mut buf);
    }
    if let Some(mut err) = child.stderr.take() {
        use std::io::Read;
        let _ = err.read_to_string(&mut buf);
    }
    auth_status_verdict(status.success(), &buf)
}

#[path = "setup_tests.rs"]
#[cfg(test)]
mod tests;
