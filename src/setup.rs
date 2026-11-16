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

/// This sidecar can itself be installed as the `devin` command
/// (GRAY_DEVIN_SUB_COMMAND pointing at a copy of us). Then `devin models
/// list` re-enters the sidecar, which resolves `devin` again and spawns
/// again: an unbounded fork bomb. Recognise that by comparing the resolved
/// path against our own executable.
fn is_self(resolved: &str) -> bool {
    let Ok(me) = std::env::current_exe() else {
        return false;
    };
    let Ok(resolved) = std::fs::canonicalize(resolved) else {
        return false;
    };
    match std::fs::canonicalize(&me) {
        Ok(me) => me == resolved,
        Err(_) => false,
    }
}

/// Depth backstop for the case where the copy cannot be recognised as us
/// (different inode, e.g. a bind mount or a container path). Each nesting
/// increments GRAY_DEVIN_SUB_DEPTH in the child env; past this many levels
/// nothing sane is happening and we refuse instead of forking again.
const MAX_DEPTH: u32 = 3;

fn current_depth() -> u32 {
    std::env::var("GRAY_DEVIN_SUB_DEPTH")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
}

fn at_depth_cap() -> bool {
    current_depth() >= MAX_DEPTH
}

fn refuse(resolved: &str) -> Option<String> {
    eprintln!(
        "gray-devin-sub: refusing to use `{resolved}` as the devin CLI: it is this \
         sidecar re-entering itself (nested {MAX_DEPTH}+ deep), which would fork \
         without bound. Point GRAY_DEVIN_SUB_COMMAND at the real Devin CLI."
    );
    None
}

pub fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Spawn `binary args`, drain both pipes on reader threads (a big listing
/// deadlocks the child on a full pipe if drained after wait), kill on
/// timeout. Returns (success, stdout, stderr).
pub fn run_captured(
    binary: &str,
    args: &[&str],
    timeout: std::time::Duration,
) -> Result<(bool, String, String), String> {
    run_captured_in(binary, args, timeout, &child_env())
}

/// [`run_captured`] with an explicit child env — account probes pass the
/// account's home (`XDG_DATA_HOME`) here.
pub fn run_captured_in(
    binary: &str,
    args: &[&str],
    timeout: std::time::Duration,
    env: &[(String, String)],
) -> Result<(bool, String, String), String> {
    use std::io::Read;
    let mut child = std::process::Command::new(binary)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .envs(env.iter().cloned())
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
        .map_err(|e| format!("{binary}: {e}"))?;
    fn drain(pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let mut s = String::new();
            let mut p = pipe;
            let _ = p.read_to_string(&mut s);
            s
        })
    }
    let out = child
        .stdout
        .take()
        .map(drain)
        .unwrap_or_else(|| std::thread::spawn(String::new));
    let err = child
        .stderr
        .take()
        .map(drain)
        .unwrap_or_else(|| std::thread::spawn(String::new));
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = out.join();
                let _ = err.join();
                return Err(format!("{binary} timed out"));
            }
            Err(e) => return Err(format!("{binary}: {e}")),
        }
    };
    Ok((
        status.map(|s| s.success()).unwrap_or(false),
        out.join().unwrap_or_default(),
        err.join().unwrap_or_default(),
    ))
}

fn env_override() -> Option<String> {
    std::env::var("GRAY_DEVIN_SUB_COMMAND")
        .ok()
        .filter(|v| !v.is_empty())
}

/// Resolve the `devin` binary: explicit override, then PATH.
pub fn resolve_command() -> Option<String> {
    let chosen = match env_override() {
        Some(v) => v,
        None => {
            let path = std::env::var_os("PATH")?;
            let mut found = None;
            for dir in std::env::split_paths(&path) {
                for name in ["devin", "devin.exe", "devin.cmd"] {
                    let p: PathBuf = dir.join(name);
                    if p.is_file() {
                        found = Some(p.to_string_lossy().into_owned());
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found?
        }
    };
    if is_self(&chosen) || at_depth_cap() {
        return refuse(&chosen);
    }
    Some(chosen)
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
    child_env_for(None)
}

/// [`child_env`] pointed at an account's credential home: the account's
/// `home` replaces `XDG_DATA_HOME` (so `devin` reads
/// `<home>/devin/credentials.toml`) and `<home>/cache` replaces
/// `XDG_CACHE_HOME` (so per-account CLI caches can't mix across
/// accounts). `None` inherits everything — the single-account path.
pub fn child_env_for(home: Option<&std::path::Path>) -> Vec<(String, String)> {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| !CONFLICTING_KEYS.contains(&k.as_str()))
        .collect();
    if let Some(h) = home {
        vars.retain(|(k, _)| k != "XDG_DATA_HOME" && k != "XDG_CACHE_HOME");
        vars.push(("XDG_DATA_HOME".into(), h.to_string_lossy().into_owned()));
        vars.push((
            "XDG_CACHE_HOME".into(),
            h.join("cache").to_string_lossy().into_owned(),
        ));
    }
    // Depth marker: incremented for every child so a copy that cannot be
    // recognised as us (different inode) still hits MAX_DEPTH and stops.
    vars.retain(|(k, _)| k != "GRAY_DEVIN_SUB_DEPTH");
    vars.push((
        "GRAY_DEVIN_SUB_DEPTH".into(),
        (current_depth() + 1).to_string(),
    ));
    vars
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

/// `devin auth status` with the same spawn hygiene as [`probe_login`],
/// returning the verdict plus the captured output for callers that can
/// parse account lines out of it (`/devin status`).
pub fn auth_status_output() -> (LoginState, Option<String>) {
    auth_status_output_in(None)
}

/// [`auth_status_output`] against an account's credential home.
pub fn auth_status_output_in(home: Option<&std::path::Path>) -> (LoginState, Option<String>) {
    let Some(binary) = resolve_command() else {
        return (LoginState::Unknown, None);
    };
    let Ok((ok, out, err)) = run_captured_in(
        &binary,
        &["auth", "status"],
        std::time::Duration::from_secs(8),
        &child_env_for(home),
    ) else {
        return (LoginState::Unknown, None);
    };
    let buf = format!("{out}{err}");
    (auth_status_verdict(ok, &buf), Some(buf))
}

/// Probe login state with `devin auth status` (8s cap; output captured and
/// never echoed — it may carry account details).
/// * exit 0 + "logged in" text → logged in.
/// * exit 0 without the phrase, or nonzero → logged out.
/// * spawn failure / timeout → `Unknown`.
pub fn probe_login() -> LoginState {
    auth_status_output().0
}

/// [`probe_login`] against an account's credential home.
pub fn probe_login_in(home: Option<&std::path::Path>) -> LoginState {
    auth_status_output_in(home).0
}

#[path = "setup_tests.rs"]
#[cfg(test)]
mod tests;
