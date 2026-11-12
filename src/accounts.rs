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
//! `{strategy, selected, accounts:[{label, home, email?, health}]}`.
//! Absent or empty `accounts` is today's single-account behavior
//! untouched: every spawn then carries no XDG override and `devin` uses
//! its own default home. `account use <label>` pins one account — the
//! pin serves until it is cleared or benched.
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
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One labeled credential home.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "AccountFile")]
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
    #[serde(default)]
    pub added_at: u64,
    pub health: Health,
    #[serde(default)]
    pub request_count: u64,
    #[serde(default)]
    pub last_used_at: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Health {
    Ready,
    Cooling { until: u64, reason: String },
    AuthFailed { until: u64, msg: String },
}

impl Health {
    fn until(&self) -> Option<u64> {
        match self {
            Health::Ready => None,
            Health::Cooling { until, .. } | Health::AuthFailed { until, .. } => Some(*until),
        }
    }

    fn message(&self) -> Option<&str> {
        match self {
            Health::Ready => None,
            Health::Cooling { reason, .. } => Some(reason.as_str()),
            Health::AuthFailed { msg, .. } => Some(msg.as_str()),
        }
    }

    pub fn benched(&self, now: u64) -> bool {
        self.until().is_some_and(|until| until > now)
    }

    fn at(&self, now: u64) -> Health {
        if self.benched(now) {
            self.clone()
        } else {
            Health::Ready
        }
    }
}

#[derive(Deserialize)]
struct AccountFile {
    label: String,
    home: PathBuf,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    health: Option<Health>,
    #[serde(default)]
    cooled_until: u64,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    added_at: u64,
    #[serde(default)]
    request_count: u64,
    #[serde(default)]
    last_used_at: u64,
}

impl From<AccountFile> for Account {
    fn from(f: AccountFile) -> Self {
        let health = match (f.health, f.cooled_until) {
            (Some(health), _) => health,
            (None, 0) => Health::Ready,
            (None, until) => Health::Cooling {
                until,
                reason: f.last_error.unwrap_or_default(),
            },
        };
        Account {
            label: f.label,
            home: f.home,
            email: f.email,
            added_at: f.added_at,
            health,
            request_count: f.request_count,
            last_used_at: f.last_used_at,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Bench {
    Limit(Duration),
    Auth(Duration),
}

impl Bench {
    fn ttl(self) -> Duration {
        match self {
            Bench::Limit(ttl) | Bench::Auth(ttl) => ttl,
        }
    }

    fn capped(self, cap: Duration) -> Bench {
        match self {
            Bench::Limit(ttl) => Bench::Limit(ttl.min(cap)),
            Bench::Auth(ttl) => Bench::Auth(ttl.min(cap)),
        }
    }
}

pub(crate) enum Outcome<'a> {
    Served,
    Failed(Bench, &'a str),
}

fn transition(health: &Health, outcome: Outcome, now: u64) -> Health {
    match outcome {
        Outcome::Served => Health::Ready,
        Outcome::Failed(bench, err) => {
            let until = now + bench.ttl().as_secs();
            if health.until().is_some_and(|held| held >= until) {
                return health.clone();
            }
            let msg: String = err.chars().take(160).collect();
            match bench {
                Bench::Limit(_) => Health::Cooling { until, reason: msg },
                Bench::Auth(_) => Health::AuthFailed { until, msg },
            }
        }
    }
}

enum Serving {
    Account(Account),
    Ambient,
    AllCooling { until: u64, count: usize },
}

fn serving(accounts: &[Account], selected: Option<&str>, strategy: Strategy, now: u64) -> Serving {
    if accounts.is_empty() {
        return Serving::Ambient;
    }
    let usable = |a: &&Account| !a.health.benched(now);
    if let Some(sel) = selected
        && let Some(a) = accounts.iter().find(|a| a.label == sel).filter(usable)
    {
        return Serving::Account(a.clone());
    }
    let next = match strategy {
        Strategy::LeastUsed => accounts
            .iter()
            .filter(usable)
            .min_by_key(|a| a.last_used_at),
        Strategy::FillFirst => accounts.iter().find(usable),
    };
    if let Some(a) = next {
        return Serving::Account(a.clone());
    }
    Serving::AllCooling {
        until: accounts
            .iter()
            .filter_map(|a| a.health.until())
            .min()
            .unwrap_or_default(),
        count: accounts.len(),
    }
}

fn cooling_reason(count: usize, until: u64, now: u64) -> String {
    format!(
        "all {count} devin-sub accounts are cooling — next ready in ~{}s",
        until.saturating_sub(now)
    )
}

#[derive(Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Strategy {
    LeastUsed,
    #[default]
    #[serde(other)]
    FillFirst,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::FillFirst => "fill_first",
            Strategy::LeastUsed => "least_used",
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    #[serde(default)]
    strategy: Strategy,
    #[serde(default)]
    selected: Option<String>,
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

pub(crate) fn with_lock<T>(file: &std::path::Path, f: impl FnOnce() -> T) -> Result<T, String> {
    let mut lock_name = file.as_os_str().to_owned();
    lock_name.push(".lock");
    let lock_path = PathBuf::from(lock_name);
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{e}"))?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file_lock = opts
        .open(&lock_path)
        .map_err(|e| format!("open {}: {e}", lock_path.display()))?;
    file_lock
        .lock()
        .map_err(|e| format!("lock {}: {e}", lock_path.display()))?;
    let out = f();
    let _ = file_lock.unlock();
    Ok(out)
}

fn read_strict(path: &std::path::Path) -> Result<State, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| {
            format!(
                "{} is unreadable ({e}); fix or remove it before changing the pool",
                path.display()
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::default()),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn unique_suffix() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "{}.{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

fn write_atomic(path: &std::path::Path, state: &State) -> Result<(), String> {
    let dir = path.parent().ok_or("accounts path has no parent")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{e}"))?;
    let bytes = serde_json::to_vec_pretty(state).map_err(|e| format!("{e}"))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}.tmp", unique_suffix()));
    let staged = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = staged {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{e}"));
    }
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

fn mutate<T>(f: impl FnOnce(&mut State) -> Result<(bool, T), String>) -> Result<T, String> {
    let path = file_path().ok_or("cannot resolve GRAY_HOME")?;
    with_lock(&path, || -> Result<T, String> {
        let mut state = read_strict(&path)?;
        let (dirty, out) = f(&mut state)?;
        if dirty {
            write_atomic(&path, &state)?;
        }
        Ok(out)
    })?
}

/// Every configured account, in priority (list) order.
pub fn configured() -> Vec<Account> {
    load().accounts
}

const PICK_PERSIST_EVERY: u64 = 60;

pub fn pick() -> Result<Option<Account>, String> {
    if load().accounts.is_empty() {
        return Ok(None);
    }
    mutate(|state| {
        let now = crate::setup::unix_secs();
        let strategy = state.strategy;
        match serving(&state.accounts, state.selected.as_deref(), strategy, now) {
            Serving::Account(a) => {
                let persist = match state.accounts.iter_mut().find(|x| x.label == a.label) {
                    Some(acc) => {
                        let due = strategy == Strategy::LeastUsed
                            || now.saturating_sub(acc.last_used_at) >= PICK_PERSIST_EVERY;
                        acc.request_count = acc.request_count.saturating_add(1);
                        if due {
                            acc.last_used_at = now;
                        }
                        due
                    }
                    None => false,
                };
                Ok((persist, Some(a)))
            }
            Serving::Ambient => Ok((false, None)),
            Serving::AllCooling { until, count } => Err(cooling_reason(count, until, now)),
        }
    })
}

/// The account a request is routed to: `fill_first` = the first account
/// in list order whose cooldown has lapsed. `Ok(None)` means no accounts
/// are configured — the legacy single-login path, where `devin` uses its
/// own default home. `Err` when accounts exist but every one is benched:
/// the message carries the earliest ready time (9router's 503 analog).
pub fn select() -> Result<Option<Account>, String> {
    pick_from(&load())
}

fn pick_from(state: &State) -> Result<Option<Account>, String> {
    let now = crate::setup::unix_secs();
    match serving(
        &state.accounts,
        state.selected.as_deref(),
        state.strategy,
        now,
    ) {
        Serving::Account(a) => Ok(Some(a)),
        Serving::Ambient => Ok(None),
        Serving::AllCooling { until, count } => Err(cooling_reason(count, until, now)),
    }
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
    let now = crate::setup::unix_secs();
    !label.is_empty()
        && configured()
            .iter()
            .any(|a| a.label == label && a.health.benched(now))
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
pub(crate) fn classify(err: &str) -> Option<Bench> {
    let e = err.to_ascii_lowercase();
    let limit = e.contains("rate limit")
        || e.contains("usage limit")
        || e.contains("usage-limit")
        || crate::session::has_status(&e, "429")
        || e.contains("resource_exhausted")
        || e.contains("too many requests")
        || e.contains("quota");
    let auth = e.contains("unauthorized")
        || e.contains("unauthenticated")
        || crate::session::has_status(&e, "401")
        || e.contains("not logged in")
        || e.contains("not authenticated")
        || e.contains("authentication required")
        || e.contains("login required");
    if limit {
        Some(Bench::Limit(retry_after(&e).unwrap_or(USAGE_LIMIT_TTL)))
    } else if auth {
        Some(Bench::Auth(retry_after(&e).unwrap_or(AUTH_TTL)))
    } else {
        None
    }
}

/// Bench `label` for `ttl`, recording the error. Sole-account pools cap
/// at [`SOLE_CAP`]; a provider-declared reset (already inside `ttl` via
/// [`classify`]) is capped the same way — a wrong parse can't bench the
/// only login for hours.
pub(crate) fn cool(label: &str, bench: Bench, err: &str) {
    if label.is_empty() {
        return;
    }
    let outcome: Result<Option<u64>, String> = mutate(|state| {
        if state.accounts.is_empty() {
            return Ok((false, None));
        }
        let bench = if state.accounts.len() == 1 {
            bench.capped(SOLE_CAP)
        } else {
            bench
        };
        let now = crate::setup::unix_secs();
        let Some(a) = state.accounts.iter_mut().find(|a| a.label == label) else {
            return Ok((false, None));
        };
        let next = transition(&a.health, Outcome::Failed(bench, err), now);
        if next == a.health {
            return Ok((false, None));
        }
        let left = next.until().unwrap_or(now).saturating_sub(now);
        a.health = next;
        Ok((true, Some(left)))
    });
    match outcome {
        Ok(Some(secs)) => {
            eprintln!("devin-sub: account {label} cooled {secs}s after upstream rejection")
        }
        Ok(None) => {}
        Err(e) => eprintln!("devin-sub: could not persist account cooldown: {e}"),
    }
}

/// A settled turn proves the account has headroom: drop any bench and
/// the recorded error.
fn clear(label: &str) {
    if label.is_empty() {
        return;
    }
    let _ = mutate(|state| {
        let now = crate::setup::unix_secs();
        let dirty = state
            .accounts
            .iter_mut()
            .find(|a| a.label == label)
            .is_some_and(|a| {
                let next = transition(&a.health, Outcome::Served, now);
                let changed = next != a.health;
                a.health = next;
                changed
            });
        Ok::<_, String>((dirty, ()))
    });
}

/// Record one turn's outcome against its serving account: success clears
/// a bench, an account-scoped failure sets one, a request-scoped failure
/// does nothing.
pub fn observe<T>(account: Option<&Account>, result: &Result<T, String>) {
    let Some(a) = account else { return };
    match result {
        Ok(_) => clear(&a.label),
        Err(e) => {
            if let Some(bench) = classify(e) {
                cool(&a.label, bench, e);
            }
        }
    }
}

/// The operator surface: `devin-sub account <verb>` and the sidecar's
/// `/devin account …`. `add` needs a real terminal — it runs the
/// provider's own `devin auth login` — and refuses otherwise.
pub fn command(args: &[String]) -> Result<String, String> {
    let json = args.iter().any(|a| a == "--json");
    let args: Vec<String> = args
        .iter()
        .filter(|a| a.as_str() != "--json")
        .cloned()
        .collect();
    match args.first().map(String::as_str) {
        Some("add") => add(&args[1..], json),
        Some("list") => list_cmd(json),
        Some("use") => use_account(args.get(1).map(String::as_str).unwrap_or(""), json),
        Some("strategy") => strategy_cmd(&args[1..], json),
        Some("remove") | Some("rm") => remove(&args[1..], json),
        Some("status") => status_cmd(json),
        Some("usage") => cmd_usage(&args[1..], json),
        _ => Err(
            "usage: devin-sub account add [--home DIR] <label> [-- login-args] | \
             list | use <label>|--clear | remove [--keep] <label> | \
             strategy [fill_first|least_used] | status | usage [<label>] \
             (any verb takes --json)"
                .to_string(),
        ),
    }
}

fn store_json(state: &State) -> Result<String, String> {
    let now = crate::setup::unix_secs();
    let accounts: Vec<Value> = state
        .accounts
        .iter()
        .map(|a| {
            let health = a.health.at(now);
            json!({
                "label": a.label,
                "home": a.home.display().to_string(),
                "email": a.email,
                "org": null,
                "added_at": a.added_at,
                "cooldown_until": health.until().unwrap_or_default(),
                "cooling": !matches!(health, Health::Ready),
                "last_error": health.message(),
                "health": health,
                "request_count": a.request_count,
                "last_used_at": a.last_used_at,
                "pinned": state.selected.as_deref() == Some(a.label.as_str()),
            })
        })
        .collect();
    let serving_json = match serving(
        &state.accounts,
        state.selected.as_deref(),
        state.strategy,
        now,
    ) {
        Serving::Account(a) => json!({"kind": "account", "label": a.label}),
        Serving::Ambient => json!({"kind": "ambient"}),
        Serving::AllCooling { until, count } => json!({
            "kind": "cooldown",
            "until": until,
            "reason": cooling_reason(count, until, now),
        }),
    };
    serde_json::to_string_pretty(&json!({
        "version": 1,
        "plugin": crate::manifest::PLUGIN_NAME,
        "provider": crate::manifest::PROVIDER_ID,
        "strategy": state.strategy.name(),
        "selected": state.selected,
        "ambient": state.accounts.is_empty(),
        "accounts": accounts,
        "serving": serving_json,
    }))
    .map_err(|e| e.to_string())
}

fn reply(message: String, json: bool) -> Result<String, String> {
    if !json {
        return Ok(message);
    }
    serde_json::to_string(&json!({"ok": true, "message": message})).map_err(|e| e.to_string())
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label != "."
        && label != ".."
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// `account add [--home DIR] <label> [-- <devin auth login args>]`: create a
/// fresh credential home and run the real `devin auth login` inside it.
fn login_into(
    binary: &str,
    label: &str,
    home: &std::path::Path,
    login_args: &[String],
) -> Result<Option<String>, String> {
    std::fs::create_dir_all(home).map_err(|e| format!("create {home:?}: {e}"))?;
    eprintln!("devin-sub: logging account {label} in via `devin auth login` (home {home:?})");
    let status = std::process::Command::new(binary)
        .arg("auth")
        .arg("login")
        .args(login_args)
        .envs(crate::setup::child_env_for(Some(home)))
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
    let (verdict, output) = crate::setup::auth_status_output_in(Some(home));
    if verdict != crate::setup::LoginState::LoggedIn {
        return Err(format!(
            "login did not verify in {home:?} — `devin auth status` there is not logged in"
        ));
    }
    Ok(output
        .as_deref()
        .and_then(|o| crate::status::field(o, "Email")))
}

fn add(args: &[String], json: bool) -> Result<String, String> {
    let mut label: Option<String> = None;
    let mut explicit: Option<PathBuf> = None;
    let mut login_args: Vec<String> = Vec::new();
    let mut it = args.iter().map(String::as_str);
    while let Some(a) = it.next() {
        match a {
            "--" => {
                login_args = it.map(str::to_string).collect();
                break;
            }
            "--home" => {
                let Some(dir) = it.next() else {
                    return Err("--home needs a directory".into());
                };
                explicit = Some(PathBuf::from(dir));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag {other:?}"));
            }
            other => {
                if label.replace(other.to_string()).is_some() {
                    return Err("usage: devin-sub account add [--home DIR] <label>".into());
                }
            }
        }
    }
    let label = label.ok_or("usage: devin-sub account add [--home DIR] <label>")?;
    if !valid_label(&label) {
        return Err(format!("bad account label: {label}"));
    }
    let gray = gray_core::paths::gray_home().ok_or("cannot resolve GRAY_HOME")?;
    let homes = gray.join("devin-sub").join("homes");
    let (home, staging) = match explicit {
        Some(dir) => (dir, None),
        None => (
            homes.join(&label),
            Some(homes.join(format!(".{label}.{}.tmp", unique_suffix()))),
        ),
    };
    if load().accounts.iter().any(|a| a.label == label) {
        return Err(format!("account {label} already exists"));
    }
    if staging.is_some() && home.read_dir().is_ok_and(|mut d| d.next().is_some()) {
        return Err(format!(
            "{home:?} already exists and is not empty; delete or move it before adding {label}"
        ));
    }
    let adopted = if staging.is_none() {
        let (verdict, output) = crate::setup::auth_status_output_in(Some(home.as_path()));
        if verdict == crate::setup::LoginState::LoggedIn {
            Some(
                output
                    .as_deref()
                    .and_then(|o| crate::status::field(o, "Email")),
            )
        } else {
            None
        }
    } else {
        None
    };
    let email = match adopted {
        Some(email) => email,
        None => {
            if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
                return Err(
                    "account add needs a terminal — run `gray devin-sub account add` in a shell"
                        .into(),
                );
            }
            let binary = crate::setup::resolve_command().ok_or(crate::setup::INSTALL_HINT)?;
            let work = staging.as_deref().unwrap_or(home.as_path());
            let login_email = match login_into(&binary, &label, work, &login_args) {
                Ok(email) => email,
                Err(e) => {
                    if let Some(s) = &staging {
                        let _ = std::fs::remove_dir_all(s);
                    }
                    return Err(e);
                }
            };
            if staging.is_some()
                && let Err(e) = std::fs::rename(work, &home)
            {
                return Err(format!(
                    "login succeeded but {home:?} cannot take it ({e}); the new login is kept at {work:?} — delete the old home or move it, then rename the new one into place"
                ));
            }
            login_email
        }
    };
    let entry = Account {
        label: label.clone(),
        home: home.clone(),
        email: email.clone(),
        added_at: crate::setup::unix_secs(),
        health: Health::Ready,
        request_count: 0,
        last_used_at: 0,
    };
    let added = match mutate(|state| {
        if state.accounts.iter().any(|a| a.label == label) {
            return Err(format!("account {label} already exists"));
        }
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
        state.accounts.push(entry.clone());
        Ok((true, entry))
    }) {
        Ok(added) => added,
        Err(e) => {
            if staging.is_some() {
                let _ = std::fs::remove_dir_all(&home);
            }
            return Err(e);
        }
    };
    reply(
        format!("account {label} added ({})", status_line(&added)),
        json,
    )
}

fn status_line(a: &Account) -> String {
    let mut s = match &a.email {
        Some(e) => format!("{e} · {}", a.home.display()),
        None => a.home.display().to_string(),
    };
    let now = crate::setup::unix_secs();
    let (label, until, reason) = match a.health.at(now) {
        Health::Ready => return s,
        Health::Cooling { until, reason } => ("cooling", until, reason),
        Health::AuthFailed { until, msg } => ("auth failed, retry in", until, msg),
    };
    s.push_str(&format!(" · {label} {}s", until.saturating_sub(now)));
    if !reason.is_empty() {
        s.push_str(&format!(
            " ({})",
            reason.chars().take(80).collect::<String>()
        ));
    }
    s
}

fn list_cmd(json: bool) -> Result<String, String> {
    if json {
        return store_json(&load());
    }
    Ok(list_text())
}

fn list_text() -> String {
    let state = load();
    if state.accounts.is_empty() {
        return "no accounts configured — single-account mode: devin's own login \
                (~/.local/share/devin). `devin-sub account add <label>` starts a pool."
            .to_string();
    }
    let now = crate::setup::unix_secs();
    let serving_now = serving(
        &state.accounts,
        state.selected.as_deref(),
        state.strategy,
        now,
    );
    let mut out = format!("accounts (strategy: {}):\n", state.strategy.name());
    for a in &state.accounts {
        let mark = match &serving_now {
            Serving::Account(s) if s.label == a.label => "*",
            _ => " ",
        };
        out.push_str(&format!("{mark} {:<14} {}\n", a.label, status_line(a)));
    }
    match serving_now {
        Serving::Account(a) => out.push_str(&format!("serving: {}", a.label)),
        Serving::AllCooling { .. } => out.push_str("serving: none (all cooling)"),
        Serving::Ambient => out.push_str("serving: ambient"),
    }
    out
}

/// `/devin status` footer + `account status`: which account serves now.
pub fn status() -> String {
    let state = load();
    match serving(
        &state.accounts,
        state.selected.as_deref(),
        state.strategy,
        crate::setup::unix_secs(),
    ) {
        Serving::Ambient => "account: single (devin's own login, ~/.local/share/devin)".to_string(),
        Serving::Account(a) => {
            let mut out = list_text();
            let (verdict, _) = crate::setup::auth_status_output_in(Some(&a.home));
            let probe = match verdict {
                crate::setup::LoginState::LoggedIn => "logged in",
                crate::setup::LoginState::LoggedOut => "NOT logged in",
                crate::setup::LoginState::Unknown => "probe inconclusive",
            };
            out.push_str(&format!(" · {probe}"));
            out
        }
        Serving::AllCooling { .. } => list_text(),
    }
}

fn use_account(label: &str, json: bool) -> Result<String, String> {
    if label.is_empty() {
        let state = load();
        let sel = state.selected.as_deref().unwrap_or("(none)");
        if json {
            return serde_json::to_string(&json!({"ok": true, "selected": state.selected}))
                .map_err(|e| e.to_string());
        }
        return Ok(format!("pinned account: {sel}"));
    }
    if matches!(label, "--clear" | "none" | "auto") {
        let selected = mutate(|state| {
            state.selected = None;
            Ok((true, state.selected.clone()))
        })?;
        if json {
            return serde_json::to_string(&json!({"ok": true, "selected": selected}))
                .map_err(|e| e.to_string());
        }
        return Ok("pin cleared — selection follows the strategy".to_string());
    }
    let selected = mutate(|state| {
        if !state.accounts.iter().any(|a| a.label == label) {
            return Err(format!("no such account: {label}"));
        }
        state.selected = Some(label.to_string());
        Ok((true, state.selected.clone()))
    })?;
    if json {
        return serde_json::to_string(&json!({"ok": true, "selected": selected}))
            .map_err(|e| e.to_string());
    }
    Ok(format!("serving account: {label}"))
}

fn strategy_cmd(args: &[String], json: bool) -> Result<String, String> {
    let Some(want) = args.first().map(String::as_str) else {
        let cur = load().strategy.name();
        if json {
            return serde_json::to_string(&json!({"ok": true, "strategy": cur}))
                .map_err(|e| e.to_string());
        }
        return Ok(format!("strategy: {cur}"));
    };
    let next = match want {
        "fill_first" => Strategy::FillFirst,
        "least_used" => Strategy::LeastUsed,
        other => {
            return Err(format!(
                "unknown strategy {other:?} — fill_first or least_used"
            ));
        }
    };
    let set = mutate(|state| {
        state.strategy = next;
        Ok((true, state.strategy.name()))
    })?;
    if json {
        return serde_json::to_string(&json!({"ok": true, "strategy": set}))
            .map_err(|e| e.to_string());
    }
    Ok(format!("strategy: {set}"))
}

fn remove(args: &[String], json: bool) -> Result<String, String> {
    let mut keep = false;
    let mut label: Option<String> = None;
    for a in args {
        if a == "--keep" {
            keep = true;
        } else if label.is_none() {
            label = Some(a.clone());
        } else {
            return Err("usage: devin-sub account remove [--keep] <label>".into());
        }
    }
    let label = label.ok_or("usage: devin-sub account remove [--keep] <label>")?;
    let a = mutate(|state| {
        let Some(i) = state.accounts.iter().position(|x| x.label == label) else {
            return Err(format!("no such account: {label}"));
        };
        let a = state.accounts.remove(i);
        if state.selected.as_deref() == Some(label.as_str()) {
            state.selected = None;
        }
        Ok((true, a))
    })?;
    let mut msg = format!("account {label:?} removed");
    if !keep
        && let Some(homes) =
            gray_core::paths::gray_home().map(|g| g.join("devin-sub").join("homes"))
        && a.home.parent() == Some(homes.as_path())
        && a.home.is_dir()
    {
        match std::fs::remove_dir_all(&a.home) {
            Ok(()) => msg.push_str(&format!(" and its credential home {}", a.home.display())),
            Err(e) => msg.push_str(&format!(
                "; its credential home {} was left behind ({e})",
                a.home.display()
            )),
        }
    } else {
        msg.push_str(&format!(" (home left at {})", a.home.display()));
    }
    reply(msg, json)
}

fn status_cmd(json: bool) -> Result<String, String> {
    if json {
        return store_json(&load());
    }
    Ok(status())
}

const USAGE_DEADLINE: Duration = Duration::from_secs(8);

fn usage_entry(label: Option<String>, outcome: Result<Value, String>) -> Value {
    match outcome {
        Ok(card) => json!({"label": label, "usage": card, "error": null}),
        Err(e) => json!({"label": label, "usage": null, "error": e}),
    }
}

enum Target {
    Ambient,
    Home(PathBuf),
}

enum Slot {
    Waiting,
    Done(Value),
}

fn usage_targets(
    accounts: &[Account],
    want: Option<&str>,
) -> Vec<(Option<String>, Result<Target, String>)> {
    if accounts.is_empty() {
        return match want {
            None => vec![(None, Ok(Target::Ambient))],
            Some(w) => vec![(
                Some(w.to_string()),
                Err(format!(
                    "no account labelled {w:?}: no pool configured, the ambient login has no label"
                )),
            )],
        };
    }
    let picked: Vec<(Option<String>, Result<Target, String>)> = accounts
        .iter()
        .filter(|a| want.is_none_or(|w| a.label == w))
        .map(|a| (Some(a.label.clone()), Ok(Target::Home(a.home.clone()))))
        .collect();
    match want {
        Some(w) if picked.is_empty() => vec![(
            Some(w.to_string()),
            Err(format!("no account labelled {w:?}")),
        )],
        _ => picked,
    }
}

fn usage_text(entry: &Value) -> String {
    let mut out = match entry.get("label").and_then(Value::as_str) {
        Some(l) => format!("{l}\n"),
        None => "ambient\n".to_string(),
    };
    if let Some(e) = entry.get("error").and_then(Value::as_str) {
        out.push_str(&format!("  error: {e}\n"));
    }
    let windows = entry.pointer("/usage/windows").and_then(Value::as_array);
    for w in windows.into_iter().flatten() {
        let name = w
            .get("label")
            .or(w.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let pct = w
            .get("used_percent")
            .and_then(Value::as_f64)
            .map_or("-".to_string(), |p| format!("{p:.0}%"));
        let reset = w.get("resets_at").and_then(Value::as_str).unwrap_or("-");
        out.push_str(&format!("  {name:<14} {pct:>5}  resets {reset}\n"));
    }
    out.trim_end().to_string()
}

fn cmd_usage(args: &[String], json: bool) -> Result<String, String> {
    let want = args.first().map(String::as_str);
    let targets = usage_targets(&load().accounts, want);
    let labels: Vec<Option<String>> = targets.iter().map(|(l, _)| l.clone()).collect();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut slots: Vec<Slot> = targets.iter().map(|_| Slot::Waiting).collect();
    for (i, (label, target)) in targets.into_iter().enumerate() {
        match target {
            Err(e) => slots[i] = Slot::Done(usage_entry(label, Err(e))),
            Ok(probe) => {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let outcome = match probe {
                        Target::Ambient => crate::usage::handle_in(None).map(|mut card| {
                            let (_, out) = crate::setup::auth_status_output_in(None);
                            if let Some(email) = out
                                .as_deref()
                                .and_then(|o| crate::status::field(o, "Email"))
                            {
                                card["email"] = serde_json::json!(email);
                            }
                            card
                        }),
                        Target::Home(home) => crate::usage::handle_in(Some(home.as_path())),
                    }
                    .map_err(|e| e.to_string());
                    let _ = tx.send((i, usage_entry(label, outcome)));
                });
            }
        }
    }
    drop(tx);
    let deadline = Instant::now() + USAGE_DEADLINE;
    while let Ok((i, entry)) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        slots[i] = Slot::Done(entry);
    }
    let entries: Vec<Value> = slots
        .into_iter()
        .zip(labels)
        .map(|(slot, label)| match slot {
            Slot::Done(entry) => entry,
            Slot::Waiting => usage_entry(
                label,
                Err(format!("no reply within {}s", USAGE_DEADLINE.as_secs())),
            ),
        })
        .collect();
    if json {
        return serde_json::to_string_pretty(&json!({
            "version": 1,
            "plugin": crate::manifest::PLUGIN_NAME,
            "provider": crate::manifest::PROVIDER_ID,
            "accounts": entries,
        }))
        .map_err(|e| e.to_string());
    }
    Ok(entries
        .iter()
        .map(usage_text)
        .collect::<Vec<_>>()
        .join("\n"))
}
