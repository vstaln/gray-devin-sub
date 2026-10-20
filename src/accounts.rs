//! Multiple `devin` CLI logins ("accounts"): each account is a labeled
//! credential *home* — a directory the sidecar exports as `XDG_DATA_HOME`
//! (and `<home>/cache` as `XDG_CACHE_HOME`) to every `devin` it spawns,
//! so the CLI resolves `<home>/devin/credentials.toml` instead of the
//! shared `~/.local/share/devin/` login. The plugin never reads, copies,
//! or forwards credential material: `account add` runs the real
//! `devin auth login` into a fresh home, and probes just ask that home's
//! `devin auth status`.
//!
//! State: `<gray home>/devin-sub/accounts.json` —
//! `{strategy, accounts:[{label, home, email?, cooled_until,
//! last_error?}]}`. Absent or empty `accounts` is today's single-account
//! behavior untouched: every spawn then carries no XDG override and
//! `devin` uses its own default home. `account use <label>` reorders the
//! list — `fill_first` selection makes list order the priority order.
//!
//! Selection (per `provider/chat`): first non-cooled account
//! (`fill_first`). An upstream usage/rate-limit or auth failure cools
//! the serving account for a TTL — a `Retry-After`-style duration parsed
//! out of the error wins over the computed default — and the next
//! request falls through to the next account. Errors that aren't
//! account-scoped (bad request, dead pipe, cancel) never cool anyone,
//! and a pool of one caps its cooldown at [`SOLE_CAP`] so a transient
//! rejection can't bench the only login for an hour.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// One labeled credential home.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Account {
    pub label: String,
    /// Exported to spawned `devin` processes as `XDG_DATA_HOME`; the CLI
    /// then reads/writes `<home>/devin/credentials.toml`.
    pub home: PathBuf,
    /// `Email:` seen at `add` time — dedupe key for repeat logins, since
    /// merging a second grant onto one upstream identity corrupts the
    /// rotated tokens of both (9router's codex scar). Advisory only:
    /// empty when the probe couldn't read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Unix seconds until which this account is benched.
    #[serde(default)]
    pub cooled_until: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl Account {
    /// Whether the account is benched right now.
    pub fn cooled(&self) -> bool {
        self.cooled_until > crate::setup::unix_secs()
    }
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    /// `fill_first` is the only strategy for now: list order is the
    /// priority order. Unknown values fall back to it.
    #[serde(default)]
    strategy: String,
    #[serde(default)]
    accounts: Vec<Account>,
}

fn file_path() -> Option<PathBuf> {
    gray_core::paths::gray_home().map(|h| h.join("devin-sub").join("accounts.json"))
}

fn load() -> State {
    let Some(path) = file_path() else {
        return State::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return State::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save(state: &State) -> Result<(), String> {
    let path = file_path().ok_or("cannot resolve GRAY_HOME")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(state).map_err(|e| format!("{e}"))?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    use std::io::Write;
    let done = opts
        .open(&tmp)
        .and_then(|mut f| f.write_all(&bytes))
        .and_then(|_| std::fs::rename(&tmp, &path));
    if done.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    done.map_err(|e| format!("{e}"))
}

/// Every configured account, in priority (list) order.
pub fn configured() -> Vec<Account> {
    load().accounts
}

/// The account a request is routed to: `fill_first` = the first account
/// in list order whose cooldown has lapsed. `Ok(None)` means no accounts
/// are configured — the legacy single-login path, where `devin` uses its
/// own default home. `Err` when accounts exist but every one is benched:
/// the message carries the earliest ready time (9router's 503 analog).
pub fn select() -> Result<Option<Account>, String> {
    let accounts = configured();
    if accounts.is_empty() {
        return Ok(None);
    }
    if let Some(a) = accounts.iter().find(|a| !a.cooled()) {
        return Ok(Some(a.clone()));
    }
    let next = accounts
        .iter()
        .map(|a| a.cooled_until)
        .min()
        .unwrap_or_default();
    let wait = next.saturating_sub(crate::setup::unix_secs());
    Err(format!(
        "all {} devin-sub accounts are cooling — next ready in ~{}s",
        accounts.len(),
        wait
    ))
}

/// The configured account with this label; `""` (the default/env login)
/// resolves to `None` — a turn that carries no account override.
pub fn resolve(label: &str) -> Option<Account> {
    if label.is_empty() {
        return None;
    }
    configured().into_iter().find(|a| a.label == label)
}

/// Whether this account label is currently benched. `""` (default login)
/// is never cooled — no account entry exists to bench.
pub fn is_cooled(label: &str) -> bool {
    !label.is_empty() && configured().iter().any(|a| a.label == label && a.cooled())
}

/// Default bench on an upstream usage/rate-limit failure.
const USAGE_LIMIT_TTL: Duration = Duration::from_secs(60 * 60);
/// Default bench on an upstream auth failure (401-ish).
const AUTH_TTL: Duration = Duration::from_secs(5 * 60);
/// A single configured account can never be benched past this: a bad
/// request would otherwise mask the real error behind "all accounts
/// locked", and transient rejections deserve a short retry, not an
/// hour-long outage.
const SOLE_CAP: Duration = Duration::from_secs(60);

/// A `Retry-After`-style wait parsed out of the error text — "try again
/// in 4 minutes", "retry after 90s", "resets in 2h". Provider-declared
/// resets win over the computed TTL. `None` when nothing parses.
fn retry_after(err_lower: &str) -> Option<Duration> {
    for key in [
        "retry after",
        "try again in",
        "again in",
        "resets in",
        "reset in",
    ] {
        let Some(at) = err_lower.find(key) else {
            continue;
        };
        let rest = &err_lower[at + key.len()..];
        let digits: String = rest
            .trim_start_matches(|c: char| !c.is_ascii_digit())
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let Ok(n) = digits.parse::<u64>() else {
            continue;
        };
        let skipped = rest.trim_start_matches(|c: char| !c.is_ascii_digit()).len();
        let unit = rest[skipped + digits.len()..].trim_start();
        let secs = if unit.starts_with('h') {
            n * 3600
        } else if unit.starts_with('m') {
            n * 60
        } else {
            n
        };
        if secs > 0 {
            return Some(Duration::from_secs(secs));
        }
    }
    None
}

/// Whether an upstream error is an *account-scoped* failure worth
/// benching the serving account for: usage/rate limits and auth
/// rejections. Anything else — a malformed request, a dead child, a
/// cancel — is request-scoped and must never cool an account (with one
/// account that would hide the real error behind "all locked").
/// Returns the bench TTL.
pub(crate) fn classify(err: &str) -> Option<Duration> {
    let e = err.to_ascii_lowercase();
    let limit = e.contains("rate limit")
        || e.contains("usage limit")
        || e.contains("usage-limit")
        || e.contains("429")
        || e.contains("resource_exhausted")
        || e.contains("too many requests")
        || e.contains("quota");
    let auth = e.contains("unauthorized")
        || e.contains("unauthenticated")
        || e.contains("401")
        || e.contains("not logged in")
        || e.contains("not authenticated")
        || e.contains("authentication required")
        || e.contains("login required");
    let base = if limit {
        USAGE_LIMIT_TTL
    } else if auth {
        AUTH_TTL
    } else {
        return None;
    };
    Some(retry_after(&e).unwrap_or(base))
}

/// Bench `label` for `ttl`, recording the error. Sole-account pools cap
/// at [`SOLE_CAP`]; a provider-declared reset (already inside `ttl` via
/// [`classify`]) is capped the same way — a wrong parse can't bench the
/// only login for hours.
pub(crate) fn cool(label: &str, ttl: Duration, err: &str) {
    if label.is_empty() {
        return;
    }
    let mut state = load();
    if state.accounts.is_empty() {
        return;
    }
    let ttl = if state.accounts.len() == 1 {
        ttl.min(SOLE_CAP)
    } else {
        ttl
    };
    let until = crate::setup::unix_secs() + ttl.as_secs();
    if let Some(a) = state.accounts.iter_mut().find(|a| a.label == label) {
        a.cooled_until = until;
        a.last_error = Some(err.chars().take(160).collect());
        if let Err(e) = save(&state) {
            eprintln!("devin-sub: could not persist account cooldown: {e}");
        }
        eprintln!(
            "devin-sub: account {label} cooled {}s after upstream rejection",
            ttl.as_secs()
        );
    }
}

/// A settled turn proves the account has headroom: drop any bench and
/// the recorded error.
fn clear(label: &str) {
    if label.is_empty() {
        return;
    }
    let mut state = load();
    let dirty = state
        .accounts
        .iter_mut()
        .find(|a| a.label == label)
        .is_some_and(|a| {
            let was = a.cooled_until != 0 || a.last_error.is_some();
            a.cooled_until = 0;
            a.last_error = None;
            was
        });
    if dirty {
        let _ = save(&state);
    }
}

/// Record one turn's outcome against its serving account: success clears
/// a bench, an account-scoped failure sets one, a request-scoped failure
/// does nothing.
pub fn observe<T>(account: Option<&Account>, result: &Result<T, String>) {
    let Some(a) = account else { return };
    match result {
        Ok(_) => clear(&a.label),
        Err(e) => {
            if let Some(ttl) = classify(e) {
                cool(&a.label, ttl, e);
            }
        }
    }
}

/// The operator surface: `devin-sub account <verb>` and the sidecar's
/// `/devin account …`. `add` needs a real terminal — it runs the
/// provider's own `devin auth login` — and refuses otherwise.
pub fn command(args: &[String]) -> Result<String, String> {
    match args.first().map(String::as_str) {
        Some("add") => add(&args[1..]),
        Some("list") => Ok(list()),
        Some("use") => use_account(args.get(1).map(String::as_str).unwrap_or("")),
        Some("remove") | Some("rm") => remove(args.get(1).map(String::as_str).unwrap_or("")),
        Some("status") => Ok(status()),
        _ => Err(
            "usage: devin-sub account add <label> [home] [-- login-args] | \
             list | use <label> | remove <label> | status"
                .to_string(),
        ),
    }
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// `account add <label> [home] [-- <devin auth login args>]`: create a
/// fresh credential home and run the real `devin auth login` inside it.
fn add(args: &[String]) -> Result<String, String> {
    let Some(label) = args.first().map(String::as_str) else {
        return Err("usage: devin-sub account add <label> [home] [-- login-args]".into());
    };
    if !valid_label(label) {
        return Err(format!("bad account label: {label}"));
    }
    let mut rest = &args[1..];
    let mut login_args: Vec<String> = Vec::new();
    if let Some(i) = rest.iter().position(|a| a == "--") {
        login_args = rest[i + 1..].to_vec();
        rest = &rest[..i];
    }
    let default_home = gray_core::paths::gray_home()
        .map(|h| h.join("devin-sub").join("homes").join(label))
        .ok_or("cannot resolve GRAY_HOME")?;
    let home = rest.first().map(PathBuf::from).unwrap_or(default_home);
    let mut state = load();
    if state.accounts.iter().any(|a| a.label == label) {
        return Err(format!("account {label} already exists"));
    }
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return Err(
            "account add needs a terminal — run `gray devin-sub account add` in a shell".into(),
        );
    }
    let binary = crate::setup::resolve_command().ok_or(crate::setup::INSTALL_HINT)?;
    std::fs::create_dir_all(&home).map_err(|e| format!("create {home:?}: {e}"))?;
    eprintln!("devin-sub: logging account {label} in via `devin auth login` (home {home:?})");
    let status = std::process::Command::new(&binary)
        .arg("auth")
        .arg("login")
        .args(&login_args)
        .envs(crate::setup::child_env_for(Some(&home)))
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| format!("{binary}: {e}"))?;
    if !status.success() {
        return Err(format!(
            "`devin auth login` exited {status}; nothing stored"
        ));
    }
    // Verify + capture the identity for dedupe: the same upstream email
    // under a second label would corrupt both logins' rotated tokens.
    let (verdict, output) = crate::setup::auth_status_output_in(Some(&home));
    if verdict != crate::setup::LoginState::LoggedIn {
        return Err(format!(
            "login did not verify in {home:?} — `devin auth status` there is not logged in"
        ));
    }
    let email = output
        .as_deref()
        .and_then(|o| crate::status::field(o, "Email"));
    if let Some(email) = email.as_deref()
        && let Some(other) = state
            .accounts
            .iter()
            .find(|a| a.email.as_deref() == Some(email))
    {
        return Err(format!(
            "{email} is already registered as account {:?} — refusing a second grant \
             on the same upstream identity",
            other.label
        ));
    }
    state.accounts.push(Account {
        label: label.to_string(),
        home,
        email,
        cooled_until: 0,
        last_error: None,
    });
    save(&state)?;
    Ok(format!(
        "account {label} added ({})",
        status_line(state.accounts.last().unwrap())
    ))
}

fn status_line(a: &Account) -> String {
    let mut s = match &a.email {
        Some(e) => format!("{e} · {}", a.home.display()),
        None => a.home.display().to_string(),
    };
    if a.cooled() {
        let left = a.cooled_until.saturating_sub(crate::setup::unix_secs());
        s.push_str(&format!(" · cooling {left}s"));
        if let Some(e) = &a.last_error {
            s.push_str(&format!(" ({})", e.chars().take(80).collect::<String>()));
        }
    }
    s
}

fn list() -> String {
    let state = load();
    if state.accounts.is_empty() {
        return "no accounts configured — single-account mode: devin's own login \
                (~/.local/share/devin). `devin-sub account add <label>` starts a pool."
            .to_string();
    }
    let serving = select().ok().flatten().map(|a| a.label);
    let mut out = "accounts (strategy: fill_first):\n".to_string();
    for a in &state.accounts {
        let mark = if serving.as_deref() == Some(a.label.as_str()) {
            "*"
        } else {
            " "
        };
        out.push_str(&format!("{mark} {:<14} {}\n", a.label, status_line(a)));
    }
    match serving {
        Some(l) => out.push_str(&format!("serving: {l}")),
        None => out.push_str("serving: none (all cooling)"),
    }
    out
}

/// `/devin status` footer + `account status`: which account serves now.
pub fn status() -> String {
    let state = load();
    if state.accounts.is_empty() {
        return "account: single (devin's own login, ~/.local/share/devin)".to_string();
    }
    let mut out = list();
    if let Ok(Some(a)) = select() {
        let (verdict, _) = crate::setup::auth_status_output_in(Some(&a.home));
        let probe = match verdict {
            crate::setup::LoginState::LoggedIn => "logged in",
            crate::setup::LoginState::LoggedOut => "NOT logged in",
            crate::setup::LoginState::Unknown => "probe inconclusive",
        };
        out.push_str(&format!(" · {probe}"));
    }
    out
}

/// `account use <label>` — make it the priority account (fill_first
/// serves the first non-cooled entry, so this moves it to the front).
fn use_account(label: &str) -> Result<String, String> {
    let mut state = load();
    let Some(i) = state.accounts.iter().position(|a| a.label == label) else {
        return Err(format!("no such account: {label}"));
    };
    let a = state.accounts.remove(i);
    state.accounts.insert(0, a);
    save(&state)?;
    Ok(format!("account {label} is now first in fill_first order"))
}

/// `account remove <label>` — drop the pool entry. The credential home
/// stays on disk (its contents are the login; deleting it is the
/// operator's call).
fn remove(label: &str) -> Result<String, String> {
    let mut state = load();
    let Some(i) = state.accounts.iter().position(|a| a.label == label) else {
        return Err(format!("no such account: {label}"));
    };
    let a = state.accounts.remove(i);
    save(&state)?;
    Ok(format!(
        "account {label} removed (home left at {}; delete it by hand if unwanted)",
        a.home.display()
    ))
}
