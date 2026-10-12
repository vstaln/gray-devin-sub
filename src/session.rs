//! A pooled `devin acp` session: one live child process + ACP session that
//! answers consecutive host turns when each request's history is a strict
//! continuation of the one the session last answered.
//!
//! Devin's prompt cache is effectively per-session: a reused session
//! already holds the whole earlier transcript upstream (measured
//! ~99.8% cached on a repeat prompt), so a continuation turn prompts with
//! only the delta — tool results and new user input. Anything else (new
//! conversation, changed system text, diverged prefix) spawns a fresh
//! `devin acp` + `session/new`.
//!
//! Pool rules: at most `MAX_LIVE` live children; a session leaves the pool
//! for the duration of its turn (exclusive use) and returns on success —
//! including a client-disconnected turn whose prompt still settled: the
//! reply was never delivered (the relay buffers a whole turn), so the
//! session records an empty echo and the next request continues it with
//! only its own tail. A failed turn whose prompt still settled upstream
//! (RPC error, refusal, a timeout the cancel-grace then settled) keeps
//! its session too: the upstream state is known, so the host's identical
//! whole-turn retry continues it — re-sending just the failed delta —
//! instead of re-billing the whole transcript on a fresh child. Only a
//! failure leaving a prompt in flight (dead wire, an unsettled cancel)
//! closes the session — it is never pooled in a state the next turn
//! can't trust. Sessions idle past `IDLE_TTL` or whose child
//! exited are reaped on every pool access and by a 60s background sweep;
//! [`shutdown`] closes everything.
//!
//! Devin's upstream prompt cache is per-ACP-session with a ~5min TTL: a
//! pooled child idles while gray's own model runs its tools, and the next
//! continuation would re-bill the whole prefix. A keepalive sweep
//! therefore sends one fixed minimal `session/prompt` on any pooled
//! session whose last upstream contact is older than `KEEPALIVE_AFTER`.
//! The session leaves the pool for the keepalive (exclusive use, exactly
//! like a real turn — no prompt can overlap on the same child), and a
//! turn that could continue a session out for a keepalive waits for it to
//! return instead of spawning fresh (a fresh session re-bills the whole
//! transcript — the very miss the keepalive exists to prevent). ACP offers
//! no fork or rewind and the cache is per-session, so the injected turn
//! does land in the child's transcript, worded so the model reads it as an
//! automated refresh rather than user input. Matching keys off
//! `absorbed` + the echo of our reply (gray's history), which the
//! keepalive never touches, so `take` still yields the same delta and the
//! next real turn simply appends after it. `last_used` (the `IDLE_TTL`
//! user-activity clock) is likewise untouched — a warmed session still
//! dies on schedule — and keepalives stop once user idleness passes
//! `keepalive_idle_max()`, where the reaper takes over anyway.
//!
//! The pool is per sidecar process, but the upstream session is not: it
//! survives in Devin's own store, `devin acp` can `session/load` it, and
//! its prompt cache stays warm for as long as the keepalives ran. So every
//! pooled session's replay state is also persisted under
//! `<gray home>/devin-sub/pool/` on each return to the pool, and a turn
//! that misses the live pool first tries [`reattach`]: an orphaned record
//! (its owner pid gone) with the same model and system text that this
//! turn strictly continues is loaded into a fresh child and prompted
//! with only the delta — a gray restart no longer re-bills every open
//! conversation's whole transcript. Shutdown therefore detaches sessions
//! that answered a prompt (kills the child, keeps the upstream session)
//! instead of deleting them; only never-prompted ones are still deleted.

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::chat::{
    CUT_NOTE, PreparedTurn, TurnResult, Usage, batch_cut_note, blocked_call, continuation,
    parse_calls_block, rand_hex, redirect_call, salvage_calls, usage_from,
};
use crate::guard::{TurnGuard, marker_name, strip_reasoning};
use crate::setup;

/// Pool bound: at most this many live `devin acp` children.
const MAX_LIVE: usize = 4;
/// A pooled session idle this long is closed on the next pool access.
/// Keepalives run until then, so this is also how long a paused
/// conversation stays cache-warm: at 15min, coming back after a coffee
/// re-billed the whole transcript on a fresh child (measured 0% cached).
pub const IDLE_TTL: Duration = Duration::from_secs(60 * 60);
/// How often the keepalive sweep checks pooled sessions for stale
/// upstream contact.
const KEEPALIVE_SWEEP: Duration = Duration::from_secs(30);
/// Fire a keepalive once the last `session/prompt` this session settled
/// is this old — comfortably under Devin's ~5min per-session prompt
/// cache TTL (plus sweep granularity), so the next continuation still
/// hits a warm prefix.
const KEEPALIVE_AFTER: Duration = Duration::from_secs(4 * 60);
/// Keep warming only while the user is plausibly mid-break. The tradeoff
/// is arithmetic: each keepalive is a billed upstream prompt, so warming
/// the full `IDLE_TTL` costs ~14 prompts per pooled session — more than
/// the single cold-cache re-bill it saves on a late return. `IDLE_TTL`
/// still governs the pool: a session past this cap stays pooled and a
/// return turn just re-warms on contact. `DEVIN_SUB_KEEPALIVE_MIN`
/// overrides (minutes; `0` disables keepalives entirely).
fn keepalive_idle_max() -> Duration {
    std::env::var("DEVIN_SUB_KEEPALIVE_MIN")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .map(|m| m * 60)
        .unwrap_or(Duration::from_secs(20 * 60))
}
/// How long after its last upstream prompt a persisted session of a dead
/// sidecar is still worth reattaching. Its owner kept it keepalive-warm
/// up to `keepalive_idle_max()` and the cache outlives the last prompt
/// by ~5min; past that a `session/load` costs what `session/new` costs (a
/// cold prefill) plus the history replay, so there is nothing to gain.
/// `DEVIN_SUB_RELOAD_MAX_MIN` overrides (minutes; `0` disables reattach).
fn reload_max_age() -> Duration {
    std::env::var("DEVIN_SUB_RELOAD_MAX_MIN")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|m| Duration::from_secs(m * 60))
        .unwrap_or(Duration::from_secs(30 * 60))
}
/// One keepalive prompt must settle inside this window; a session that
/// can't is closed as untrusted, the same rule as a failed real turn.
const KEEPALIVE_WINDOW: Duration = Duration::from_secs(60);
/// After a keepalive fails on the account's usage/rate limit, no session
/// is warmed for this long: every retry is another request against the
/// same limit (observed: 4-minute retries for hours, every one refused),
/// and a real turn that arrives meanwhile re-warms on contact anyway.
const KEEPALIVE_COOLDOWN: Duration = Duration::from_secs(10 * 60);
/// Fixed minimal keepalive prompt: refreshes the upstream cache for a
/// near-zero reply (ACP's `session/prompt` takes no output cap). It stays
/// in the model's context, so it says what it is: a bare "." reads as a
/// user nudge and invites a full reply.
const KEEPALIVE_TEXT: &str = "[automated cache refresh, not from the user: reply with only \"ok\"]";
/// How often an ACP wait wakes to check for a relay client disconnect.
const CANCEL_POLL: Duration = Duration::from_secs(1);
/// Internal error when the relay client went away mid-turn.
const CLIENT_GONE: &str = "client disconnected";
/// A real prompt settling `end_turn` with zero output is the silent
/// usage-limit rejection ACP agents can return instead of an error (the
/// turn "succeeds" with an empty reply the host can't tell from a real
/// answer). Surfaced as a failure so the host retries; no transcript
/// content in the message.
const EMPTY_END_TURN: &str = "devin acp returned an empty end_turn with no output — likely a usage-limit rejection; session kept pooled";

/// Deny list written into both the project config and the user config the
/// child sees: every native capability Devin exposes outside `ASK_RULES`.
const DENY_RULES: &[&str] = &[
    "Fetch(domain:*)",
    "mcp__*",
    // Asks the user for directory access — impossible over the relay
    // (session/request_permission is always cancelled), so deny it up
    // front rather than let the call surface as a native__ dead end.
    "request_scope",
];

/// Added when the host surface has no `web_search`: deny the native tool
/// upstream so the model gets Devin's own "denied by rule" answer instead
/// of reaching it, the funnel catching the tool_call update, and the host
/// returning a made-up `native__web_search_not_enabled_for_devin` error
/// the model retries past. When the host DOES advertise web_search the
/// rule stays off and the funnel redirect keeps serving the call.
const DENY_SEARCH_RULE: &str = "web_search";

const ASK_RULES: &[&str] = &["read", "edit", "grep", "glob", "exec"];

const BATCH_QUIET: Duration = Duration::from_millis(250);

/// Process-wide staging: one scratch cwd + generated user config shared by
/// every pooled child, created lazily on first spawn. The dirs are STABLE
/// per surface variant, under the user's own gray home (never a
/// predictable shared /tmp path): the child's cwd is part of Devin's
/// system prompt, so a per-process random dir would give every
/// reattached session (see [`reattach`]) a different prompt prefix and a
/// cold cache — measured 0 cached tokens on a reload against ~99% on an
/// in-process reuse. Only without a gray home does a private tempdir
/// stand in, removed by [`shutdown`].
struct AcpStage {
    work: std::path::PathBuf,
    cfg: std::path::PathBuf,
    _tmp: Option<tempfile::TempDir>,
}

static STAGE: [OnceLock<AcpStage>; 2] = [OnceLock::new(), OnceLock::new()];

impl AcpStage {
    fn make(deny_search: bool) -> Result<Self, String> {
        match gray_core::paths::gray_home() {
            Some(home) => {
                let base = home.join("devin-sub").join("stage").join(if deny_search {
                    "search-denied"
                } else {
                    "search-open"
                });
                Self::make_in(base.join("work"), base.join("cfg"), deny_search, None)
            }
            None => {
                let tmp = tempfile::Builder::new()
                    .prefix("devin-sub-stage-")
                    .tempdir()
                    .map_err(|e| format!("staging dir: {e}"))?;
                let root = tmp.path().to_path_buf();
                Self::make_in(root.join("work"), root.join("cfg"), deny_search, Some(tmp))
            }
        }
    }

    fn make_in(
        work: std::path::PathBuf,
        cfg: std::path::PathBuf,
        deny_search: bool,
        tmp: Option<tempfile::TempDir>,
    ) -> Result<Self, String> {
        // Project-level deny: verified to actually block native read/exec
        // (the user-config deny alone does not).
        let proj = work.join(".devin");
        std::fs::create_dir_all(&proj).map_err(|e| format!("staging dir: {e}"))?;
        std::fs::create_dir_all(&cfg).map_err(|e| format!("staging dir: {e}"))?;
        let mut deny: Vec<&str> = DENY_RULES.to_vec();
        if deny_search {
            deny.push(DENY_SEARCH_RULE);
        }
        write_atomic(
            &proj.join("config.json"),
            json!({"permissions": {"deny": deny, "ask": ASK_RULES}})
                .to_string()
                .as_bytes(),
        )
        .map_err(|e| format!("staging config: {e}"))?;
        // User config: no MCP servers, rules, skills, subagents, auto-update
        // or notifications — but credentials still load.
        write_atomic(
            &cfg.join("config.json"),
            json!({
                "version": 1,
                "subagents_enabled": false,
                "auto_update": false,
                "notify": "never",
                "read_config_from": {
                    "agents_standard": false, "cursor": false, "windsurf": false,
                    "claude": false, "copilot": false, "opencode": false, "zed": false,
                },
                "permissions": {"deny": deny, "allow": [], "ask": ASK_RULES},
            })
            .to_string()
            .as_bytes(),
        )
        .map_err(|e| format!("staging config: {e}"))?;
        Ok(Self {
            work,
            cfg,
            _tmp: tmp,
        })
    }

    /// Stable dirs are shared with every other sidecar process and stay;
    /// only a private tempdir stand-in is removed.
    fn remove(&self) {
        if self._tmp.is_some() {
            let _ = std::fs::remove_dir_all(&self.work);
            let _ = std::fs::remove_dir_all(&self.cfg);
        }
    }
}

/// Write via a sibling temp file + rename: several sidecars may stage the
/// same (identical) config at once, and a child must never read a
/// half-written one.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("stage"),
        std::process::id()
    ));
    let done = std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, path));
    if done.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    done
}

/// The stage for this surface variant — pooled children group by whether
/// native web_search is denied, since the generated config bakes it in.
fn stage(deny_search: bool) -> Result<&'static AcpStage, String> {
    let slot = &STAGE[deny_search as usize];
    if let Some(s) = slot.get() {
        return Ok(s);
    }
    let made = AcpStage::make(deny_search)?;
    Ok(slot.get_or_init(|| made))
}

/// One live `devin acp` child + ACP session, with the replay state needed
/// to recognize and answer its strict continuation turns.
pub struct LiveSession {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    next_id: u64,
    pub(crate) session_id: String,
    pub(crate) native_model: String,
    /// Funnel system text (instructions + contract incl. tool specs) the
    /// first prompt carried; a reused session only matches a request whose
    /// prepared system is identical.
    system: String,
    /// The exact `input` array of the last request this session answered.
    pub(crate) absorbed: Vec<Value>,
    /// Call ids emitted in that answer, in order.
    reply_call_ids: Vec<String>,
    /// The (trimmed) text emitted in that answer.
    reply_text: String,
    /// User-activity clock for the `IDLE_TTL` reaper: bumped by `absorb`
    /// only. A keepalive must never touch it or a warmed session would
    /// never die.
    last_used: Instant,
    /// Last `session/prompt` this session settled upstream, real or
    /// keepalive: the clock `KEEPALIVE_AFTER` measures. `None` until the
    /// first real prompt lands — the sweep never warms a session that
    /// hasn't answered one yet.
    last_prompt_at: Option<Instant>,
    /// Keepalive prompts this session absorbed; folded into the debug
    /// trace only — never into `absorbed`, the reply echo, or host usage
    /// accounting.
    keepalive_turns: u32,
    /// A `session/prompt` whose terminal response never arrived: the
    /// upstream turn may still be live, so the session can't be pooled —
    /// a new prompt would interleave with it and late chunks would fold
    /// into the wrong turn. Cleared the moment the prompt's result (or
    /// error) comes off the wire, which is what makes a failed turn's
    /// session poolable: its upstream state is then known.
    pub(crate) prompt_in_flight: bool,
    // Per-prompt state, reset in `send_prompt`.
    text: String,
    /// A non-message `session/update` arrived since the last message chunk:
    /// the upstream segment ended, so the next chunk opens a new one and
    /// gets a paragraph break — otherwise the segments fuse ("context.Now").
    text_boundary: bool,
    thought: String,
    /// Native tool_call notifications redirected onto bash (name, args-json).
    redirected: Vec<(String, String)>,
    dropped_siblings: usize,
    prior_dropped: usize,
    seen_calls: HashSet<String>,
    pending_perms: Vec<Value>,
    batch_quiet: Option<Instant>,
    batch_flushed: bool,
    guard: TurnGuard,
    prior_cut: bool,
    /// Last `usage_update` snapshot for the in-flight prompt.
    usage: Usage,
    names: Vec<String>,
    /// This prompt is the fixed keepalive: a near-empty `end_turn` is a
    /// fine answer to it, never a silent drop.
    keepalive_prompt: bool,
    /// A `session/load` is replaying history: its `session/update`s are
    /// past turns, not this prompt's stream, and must not fold into the
    /// per-prompt state or trigger the native-call cancel.
    loading: bool,
}

/// Replay state of a pooled session as persisted for [`reattach`]:
/// everything `find` keys off, plus the two clocks as unix seconds and
/// the sidecar pid that held the live child when it was written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Persisted {
    pub session_id: String,
    pub native_model: String,
    pub system: String,
    pub absorbed: Vec<Value>,
    pub reply_call_ids: Vec<String>,
    pub reply_text: String,
    pub last_used_unix: u64,
    pub last_prompt_unix: u64,
    pub owner_pid: u32,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// `<gray home>/devin-sub/pool`, the persisted replay records.
fn pool_dir() -> Option<PathBuf> {
    gray_core::paths::gray_home().map(|h| h.join("devin-sub").join("pool"))
}

fn persisted_path(session_id: &str) -> Option<PathBuf> {
    let safe: String = session_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .collect();
    if safe.is_empty() {
        return None;
    }
    pool_dir().map(|d| d.join(format!("{safe}.json")))
}

/// Write (0600, atomic rename) the replay record of a session that has
/// answered at least one real prompt; a never-prompted session has no
/// upstream transcript worth coming back to. Best effort: a failed write
/// only costs the reattach it would have enabled.
fn persist(s: &LiveSession) {
    let Some(record) = s.persisted() else { return };
    let Some(path) = persisted_path(&record.session_id) else {
        return;
    };
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let Ok(bytes) = serde_json::to_vec(&record) else {
        return;
    };
    let tmp = dir.join(format!(".{}.{}.tmp", record.session_id, std::process::id()));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let written = opts
        .open(&tmp)
        .and_then(|mut f| f.write_all(&bytes))
        .and_then(|_| std::fs::rename(&tmp, &path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn unpersist(session_id: &str) {
    if let Some(path) = persisted_path(session_id) {
        let _ = std::fs::remove_file(path);
    }
}

/// Every parseable record in the pool dir; unparseable files are removed.
fn read_persisted() -> Vec<Persisted> {
    let Some(dir) = pool_dir() else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice::<Persisted>(&b).ok())
        {
            Some(p) => out.push(p),
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    out
}

/// Delete records older than [`reload_max_age`] (plus the sweep slack a
/// dead owner can't provide). Called once per sidecar start.
pub fn prune_persisted() {
    let cutoff = unix_now().saturating_sub(reload_max_age().as_secs() + IDLE_TTL.as_secs());
    for p in read_persisted() {
        if p.last_prompt_unix < cutoff {
            unpersist(&p.session_id);
        }
    }
}

/// Whether `pid` is a running process. Linux `/proc` probe; elsewhere
/// unknown counts as dead and Devin's own session lock is the backstop
/// (a load of a session held by a live sidecar fails `session_locked`).
fn pid_alive(pid: u32) -> bool {
    pid == std::process::id() || Path::new(&format!("/proc/{pid}")).exists()
}

/// Persisted sessions this turn could reattach to, newest upstream
/// contact first, each with the delta it would be prompted with: records
/// no live sidecar owns (`alive(owner_pid)` false), younger than
/// `max_age`, with this turn's model and system text, that this turn's
/// history strictly continues. Pure over its inputs so tests can drive it.
pub(crate) fn reload_candidates(
    records: Vec<Persisted>,
    turn: &PreparedTurn,
    now_unix: u64,
    max_age: Duration,
    alive: &dyn Fn(u32) -> bool,
) -> Vec<(Persisted, String)> {
    let mut out: Vec<(Persisted, String)> = records
        .into_iter()
        .filter(|p| !alive(p.owner_pid))
        .filter(|p| now_unix.saturating_sub(p.last_prompt_unix) <= max_age.as_secs())
        .filter(|p| p.native_model == turn.native_model && p.system == turn.system)
        .filter_map(|p| {
            let delta =
                continuation(&p.absorbed, &p.reply_call_ids, &p.reply_text, &turn.input).ok()?;
            Some((p, delta))
        })
        .collect();
    out.sort_by_key(|e| std::cmp::Reverse(e.0.last_prompt_unix));
    out
}

/// Reattach to an orphaned upstream session this turn strictly continues:
/// a fresh child, `session/load`, and the live session plus the delta to
/// prompt with — the same shape `take` returns for a pooled reuse. `None`
/// when no record qualifies or none loads; a record whose session is gone
/// upstream is removed, one that is merely locked (a sidecar whose pid
/// this host can't see) is left for its owner.
pub fn reattach(
    turn: &PreparedTurn,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Option<(LiveSession, String)> {
    let max_age = reload_max_age();
    if max_age.is_zero() {
        return None;
    }
    let candidates = reload_candidates(read_persisted(), turn, unix_now(), max_age, &pid_alive);
    for (record, delta) in candidates {
        if cancel.load(Ordering::SeqCst) {
            return None;
        }
        let stage = stage(!turn.names.iter().any(|n| n == "web_search")).ok()?;
        let mut s = spawn_child(turn, stage).ok()?;
        match s.load(&record.session_id, &stage.work, deadline, cancel) {
            Ok(()) => {
                let now = unix_now();
                let back = |t: u64| {
                    Instant::now()
                        .checked_sub(Duration::from_secs(now.saturating_sub(t)))
                        .unwrap_or_else(Instant::now)
                };
                s.absorbed = record.absorbed;
                s.reply_call_ids = record.reply_call_ids;
                s.reply_text = record.reply_text;
                s.last_used = back(record.last_used_unix);
                s.last_prompt_at = Some(back(record.last_prompt_unix));
                // Ours now: a concurrent sidecar must see a live owner.
                persist(&s);
                return Some((s, delta));
            }
            Err(e) => {
                let gone = e.to_lowercase().contains("not found");
                crate::chat::pool_event(
                    if gone { "reload:gone" } else { "reload:failed" },
                    &record.session_id,
                );
                if gone {
                    unpersist(&record.session_id);
                }
                s.abandon();
            }
        }
    }
    None
}

/// A session out of the pool for a keepalive: enough to tell whether a
/// turn could continue it.
struct Warming {
    session_id: String,
    native_model: String,
    system: String,
}

#[derive(Default)]
struct Pool {
    live: Vec<LiveSession>,
    warming: Vec<Warming>,
}

impl Pool {
    /// Whether a session out for a keepalive could answer `turn`: same
    /// model and system text, a cheap stand-in for the full continuation
    /// check. An unrelated conversation sharing both just waits out one
    /// keepalive (seconds) before spawning fresh.
    fn warming_for(&self, turn: &PreparedTurn) -> bool {
        self.warming
            .iter()
            .any(|w| w.native_model == turn.native_model && w.system == turn.system)
    }
}

/// The process-wide session pool.
static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
/// Signalled whenever a keepalive ends, so a waiting `take` re-checks.
static POOL_CHANGED: Condvar = Condvar::new();

fn locked_pool() -> MutexGuard<'static, Pool> {
    let pool = POOL.get_or_init(|| Mutex::new(Pool::default()));
    // Tiny reaper: closes idle/dead sessions even when no turn is running.
    static REAPER: Once = Once::new();
    REAPER.call_once(|| {
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(Duration::from_secs(60));
                let dead = {
                    let Some(pool) = POOL.get() else { continue };
                    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
                    reap(&mut pool.live)
                };
                for s in dead {
                    s.close();
                }
            }
        });
        // Cache keepalive: the upstream prompt cache outlives a turn only
        // ~5min, so pooled children get a minimal refresh prompt while
        // they wait out gray's own tool loop (see `keepalive_sweep`).
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(KEEPALIVE_SWEEP);
                keepalive_sweep();
            }
        });
    });
    pool.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pull dead/idle sessions out of a locked pool (caller closes them after
/// releasing the lock — `close` can block a few seconds on `session/delete`).
fn reap(pool: &mut Vec<LiveSession>) -> Vec<LiveSession> {
    let mut dead = Vec::new();
    let mut i = 0;
    while i < pool.len() {
        let idle = pool[i].last_used.elapsed() > IDLE_TTL;
        let exited = !matches!(pool[i].child.try_wait(), Ok(None));
        if idle || exited {
            let why = if exited {
                "reap:child_exited"
            } else {
                "reap:idle"
            };
            crate::chat::pool_event(why, &pool[i].session_id);
            dead.push(pool.remove(i));
        } else {
            i += 1;
        }
    }
    dead
}

/// The first pooled session this turn is a strict continuation of (same
/// native model, same system text, `continuation` yields a delta): its
/// index plus the delta, or on a miss the deepest reason any candidate
/// reached (`no_session` when the pool was empty).
fn find(live: &[LiveSession], turn: &PreparedTurn) -> Result<(usize, String), &'static str> {
    let mut reason = "no_session";
    let mut depth = 0;
    for (i, s) in live.iter().enumerate() {
        let (miss, d) = if s.native_model != turn.native_model {
            ("model", 1)
        } else if s.system != turn.system {
            ("system", 2)
        } else {
            match continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &turn.input) {
                Ok(delta) => return Ok((i, delta)),
                Err(m) => (
                    m,
                    match m {
                        "prefix" => 3,
                        "echo" => 4,
                        _ => 5,
                    },
                ),
            }
        };
        if d > depth {
            depth = d;
            reason = miss;
        }
    }
    Err(reason)
}

/// Take the pooled session this turn continues (see [`find`]), plus the
/// delta to prompt with. While a session that could answer it is out for
/// a keepalive, wait for it to come back — up to `KEEPALIVE_WINDOW`, or
/// until `cancel` — rather than miss and re-bill the whole transcript on a
/// fresh child. On a miss the second item is the reason.
pub fn take(
    turn: &PreparedTurn,
    cancel: &AtomicBool,
) -> (Option<(LiveSession, String)>, &'static str) {
    let deadline = Instant::now() + KEEPALIVE_WINDOW;
    let mut dead = Vec::new();
    let result = {
        let mut pool = locked_pool();
        loop {
            dead.extend(reap(&mut pool.live));
            let reason = match find(&pool.live, turn) {
                Ok((i, delta)) => break (Some((pool.live.remove(i), delta)), "reuse"),
                Err(reason) => reason,
            };
            let left = deadline.saturating_duration_since(Instant::now());
            if !pool.warming_for(turn) || cancel.load(Ordering::SeqCst) {
                break (None, reason);
            }
            if left.is_zero() {
                break (None, "keepalive_busy");
            }
            pool = POOL_CHANGED
                .wait_timeout(pool, left.min(CANCEL_POLL))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    };
    for s in dead {
        s.close();
    }
    result
}

pub(crate) fn prefix_divergence(turn: &PreparedTurn) -> Option<String> {
    let pool = locked_pool();
    let mut best: Option<(usize, usize)> = None;
    for s in &pool.live {
        if s.native_model != turn.native_model || s.system != turn.system {
            continue;
        }
        if crate::chat::continuation(&s.absorbed, &s.reply_call_ids, &s.reply_text, &turn.input)
            != Err("prefix")
        {
            continue;
        }
        let diverge_at = s
            .absorbed
            .iter()
            .zip(turn.input.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| s.absorbed.len().min(turn.input.len()));
        if best.map(|(d, _)| diverge_at > d).unwrap_or(true) {
            best = Some((diverge_at, s.absorbed.len()));
        }
    }
    let (diverge_at, absorbed) = best?;
    let role = turn
        .input
        .get(diverge_at)
        .map(|v| {
            v.get("role")
                .and_then(Value::as_str)
                .or_else(|| v.get("type").and_then(Value::as_str))
                .unwrap_or("?")
        })
        .unwrap_or("end");
    Some(format!(
        "diverge_at={diverge_at},absorbed={absorbed},input={},role={role}",
        turn.input.len()
    ))
}

pub(crate) fn persisted_rejections(turn: &PreparedTurn) -> String {
    let records = read_persisted();
    if records.is_empty() {
        return "persisted=0".to_string();
    }
    let now = unix_now();
    let max_age = reload_max_age().as_secs();
    let (mut owner_alive, mut too_old, mut model, mut system, mut no_continuation) =
        (0, 0, 0, 0, 0);
    for p in &records {
        if pid_alive(p.owner_pid) {
            owner_alive += 1;
        } else if now.saturating_sub(p.last_prompt_unix) > max_age {
            too_old += 1;
        } else if p.native_model != turn.native_model {
            model += 1;
        } else if p.system != turn.system {
            system += 1;
        } else if crate::chat::continuation(
            &p.absorbed,
            &p.reply_call_ids,
            &p.reply_text,
            &turn.input,
        )
        .is_err()
        {
            no_continuation += 1;
        }
    }
    format!(
        "persisted={},owner_alive={owner_alive},too_old={too_old},model={model},system={system},no_continuation={no_continuation}",
        records.len()
    )
}

/// Return a session to the pool after a successful turn or keepalive,
/// evicting the least-recently-used session past `MAX_LIVE`.
pub fn give_back(s: LiveSession) {
    persist(&s);
    let mut dead = {
        let mut pool = locked_pool();
        let mut dead = reap(&mut pool.live);
        pool.warming.retain(|w| w.session_id != s.session_id);
        pool.live.push(s);
        if pool.live.len() > MAX_LIVE {
            let lru = pool
                .live
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            crate::chat::pool_event("evict:max_live", &pool.live[lru].session_id);
            dead.push(pool.live.remove(lru));
        }
        dead
    };
    POOL_CHANGED.notify_all();
    for s in dead.drain(..) {
        s.close();
    }
}

/// Whether an upstream error is the account's usage/rate limit — the
/// one failure a retry can only prolong.
fn is_usage_limit(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("rate limit") || e.contains("usage limit") || e.contains("429")
}

/// When keepalives resume after a usage-limit failure (`None` = not paused).
static KEEPALIVE_PAUSED_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

fn keepalive_paused() -> bool {
    KEEPALIVE_PAUSED_UNTIL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some_and(|until| Instant::now() < until)
}

/// Keepalive predicate, kept field-level so tests can drive it without
/// a live child: the session settled ≥1 real prompt, its last upstream
/// contact is stale past `KEEPALIVE_AFTER`, the user is still active
/// enough for a warm cache to pay off, and the child is alive (dead
/// children are the reaper's job, not the sweep's).
fn keepalive_due(last_prompt_at: Option<Instant>, last_used: Instant, alive: bool) -> bool {
    alive
        && last_prompt_at.is_some_and(|t| t.elapsed() > KEEPALIVE_AFTER)
        && last_used.elapsed() <= keepalive_idle_max()
}

/// Remove the first keepalive-due session from the pool. Out of the pool
/// is exclusive use — exactly the serialization `take` gives a real
/// turn — so no real prompt can overlap a keepalive on the same child; it
/// is listed as warming so a host turn arriving meanwhile waits for it.
fn take_keepalive() -> Option<LiveSession> {
    if keepalive_paused() {
        return None;
    }
    let pool = POOL.get()?;
    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
    let due: Vec<bool> = pool
        .live
        .iter_mut()
        .map(|s| {
            keepalive_due(
                s.last_prompt_at,
                s.last_used,
                matches!(s.child.try_wait(), Ok(None)),
            )
        })
        .collect();
    let i = (0..pool.live.len()).find(|&i| {
        due[i] && {
            let s = &pool.live[i];
            !pool.live.iter().enumerate().any(|(j, o)| {
                j != i
                    && o.native_model == s.native_model
                    && o.system == s.system
                    && o.last_prompt_at > s.last_prompt_at
            })
        }
    })?;
    let s = pool.live.remove(i);
    pool.warming.push(Warming {
        session_id: s.session_id.clone(),
        native_model: s.native_model.clone(),
        system: s.system.clone(),
    });
    Some(s)
}

/// Warm every due session, one at a time. A settled keepalive returns
/// through `give_back` so `MAX_LIVE` still bounds the pool; a failure
/// that leaves a prompt in flight closes the session — it is never
/// pooled in a state the next turn can't trust. Either way it stops
/// being listed as warming.
fn keepalive_sweep() {
    while let Some(mut s) = take_keepalive() {
        // A failed keepalive whose prompt still settled (RPC error) costs
        // the session nothing — real-turn matching state is untouched —
        // so it goes back to the pool like a settled real-turn failure.
        // Only a prompt left in flight closes the child.
        let failed = s.keepalive().err();
        if let Some(e) = failed.as_deref().filter(|e| is_usage_limit(e)) {
            *KEEPALIVE_PAUSED_UNTIL
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now() + KEEPALIVE_COOLDOWN);
            crate::chat::pool_event(
                &format!("keepalive:paused:{}m", KEEPALIVE_COOLDOWN.as_secs() / 60),
                &s.session_id,
            );
            eprintln!(
                "devin-sub: keepalives paused {}s after usage limit: {e}",
                KEEPALIVE_COOLDOWN.as_secs()
            );
        }
        if failed.is_none() || !s.prompt_in_flight {
            give_back(s);
        } else {
            locked_pool()
                .warming
                .retain(|w| w.session_id != s.session_id);
            POOL_CHANGED.notify_all();
            s.close();
        }
    }
}

/// Close every pooled session and remove the process-wide stage dirs.
/// Called on `plugin/shutdown` and when the host's stdin goes away.
pub fn shutdown() {
    let dead = {
        let mut pool = locked_pool();
        std::mem::take(&mut pool.live)
    };
    for s in dead {
        // A session that answered a prompt is persisted and worth coming
        // back to from the next sidecar; one that never did has nothing
        // upstream and is deleted as before.
        if s.last_prompt_at.is_some() && !s.prompt_in_flight {
            persist(&s);
            s.abandon();
        } else {
            s.close();
        }
    }
    for slot in &STAGE {
        if let Some(stage) = slot.get() {
            stage.remove();
        }
    }
}

/// Spawn a fresh `devin acp` child and drive initialize + session/new.
/// `cancel` aborts the handshake when the relay client is already gone.
pub fn spawn(
    turn: &PreparedTurn,
    deadline: Instant,
    cancel: &AtomicBool,
) -> Result<LiveSession, String> {
    let stage = stage(!turn.names.iter().any(|n| n == "web_search"))?;
    let mut s = spawn_child(turn, stage)?;
    if let Err(e) = s.handshake(&stage.work, deadline, cancel) {
        s.close();
        return Err(e);
    }
    Ok(s)
}

/// Spawn a `devin acp` child on `stage` with the reader threads wired and
/// an empty replay state; no ACP traffic yet (`handshake` or `load` next).
fn spawn_child(turn: &PreparedTurn, stage: &AcpStage) -> Result<LiveSession, String> {
    let binary = setup::resolve_command().ok_or_else(|| setup::INSTALL_HINT.to_string())?;
    let mut child = Command::new(&binary)
        .args([
            "--config",
            &stage.cfg.join("config.json").to_string_lossy(),
            "acp",
            "--model",
            &turn.native_model,
        ])
        .current_dir(&stage.work)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(setup::child_env())
        // A turn must never pop a browser out of a stale login.
        .env("BROWSER", "/bin/true")
        .env("DISPLAY", "")
        .env("WAYLAND_DISPLAY", "")
        .spawn()
        .map_err(|_| setup::INSTALL_HINT.to_string())?;

    // stderr → last ~20 lines, redacted, for error detail only.
    let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
    if let Some(err) = child.stderr.take() {
        let tail = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                if let Ok(mut t) = tail.lock() {
                    if t.len() >= 20 {
                        t.pop_front();
                    }
                    t.push_back(redact(&line));
                }
            }
        });
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "native stdout unavailable".to_string())?;
    let (tx, rx) = std::sync::mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(line)
                && tx.send(v).is_err()
            {
                break;
            }
        }
    });
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "native stdin unavailable".to_string())?;

    let s = LiveSession {
        child,
        stdin,
        rx,
        stderr_tail,
        next_id: 0,
        session_id: String::new(),
        native_model: turn.native_model.clone(),
        system: turn.system.clone(),
        absorbed: Vec::new(),
        reply_call_ids: Vec::new(),
        reply_text: String::new(),
        last_used: Instant::now(),
        last_prompt_at: None,
        keepalive_turns: 0,
        prompt_in_flight: false,
        text: String::new(),
        thought: String::new(),
        text_boundary: false,
        redirected: Vec::new(),
        dropped_siblings: 0,
        prior_dropped: 0,
        seen_calls: HashSet::new(),
        pending_perms: Vec::new(),
        batch_quiet: None,
        batch_flushed: false,
        guard: TurnGuard::default(),
        prior_cut: false,
        usage: Usage::default(),
        names: turn.names.clone(),
        keepalive_prompt: false,
        loading: false,
    };
    Ok(s)
}

impl LiveSession {
    /// initialize → session/new; leaves the session ready for prompts.
    fn handshake(
        &mut self,
        work: &std::path::Path,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        self.call(
            "initialize",
            json!({"protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false},
                    "terminal": false}}),
            Duration::from_secs(60),
            deadline,
            cancel,
        )?;
        let new = self.call(
            "session/new",
            json!({"cwd": work.to_string_lossy(), "mcpServers": []}),
            Duration::from_secs(60),
            deadline,
            cancel,
        )?;
        self.session_id = new
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok(())
    }

    /// initialize → session/load of an orphaned upstream session. The
    /// agent replays the whole history as `session/update`s before it
    /// answers; `loading` keeps them out of the per-prompt state.
    fn load(
        &mut self,
        session_id: &str,
        work: &Path,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        self.call(
            "initialize",
            json!({"protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false},
                    "terminal": false}}),
            Duration::from_secs(60),
            deadline,
            cancel,
        )?;
        self.loading = true;
        let loaded = self.call(
            "session/load",
            json!({"sessionId": session_id, "cwd": work.to_string_lossy(), "mcpServers": []}),
            Duration::from_secs(120),
            deadline,
            cancel,
        );
        self.loading = false;
        loaded?;
        self.session_id = session_id.to_string();
        Ok(())
    }

    /// The replay record of a session that answered a real prompt.
    fn persisted(&self) -> Option<Persisted> {
        let last_prompt = self.last_prompt_at?;
        let now = unix_now();
        Some(Persisted {
            session_id: self.session_id.clone(),
            native_model: self.native_model.clone(),
            system: self.system.clone(),
            absorbed: self.absorbed.clone(),
            reply_call_ids: self.reply_call_ids.clone(),
            reply_text: self.reply_text.clone(),
            last_used_unix: now.saturating_sub(self.last_used.elapsed().as_secs()),
            last_prompt_unix: now.saturating_sub(last_prompt.elapsed().as_secs()),
            owner_pid: std::process::id(),
        })
    }

    /// Drain stale wire traffic, reset per-prompt state and send the
    /// `session/prompt` request. Late chunks from a previous prompt must
    /// not leak into this one: queued server requests are answered
    /// (permissions cancel, the rest refused), stale notifications and
    /// responses are dropped.
    pub(crate) fn send_prompt(
        &mut self,
        text: &str,
        images: &[Value],
        names: &[String],
    ) -> Result<u64, String> {
        while let Ok(v) = self.rx.try_recv() {
            if v.get("method").is_some() && v.get("id").is_some() {
                self.on_request(&v)?;
            }
        }
        self.text.clear();
        self.text_boundary = false;
        self.thought.clear();
        self.redirected.clear();
        self.dropped_siblings = 0;
        self.seen_calls.clear();
        self.pending_perms.clear();
        self.batch_quiet = None;
        self.batch_flushed = false;
        self.guard = TurnGuard::default();
        self.usage = Usage::default();
        self.names = names.to_vec();
        self.keepalive_prompt = text == KEEPALIVE_TEXT;
        let mut prompt = vec![json!({"type": "text", "text": text})];
        prompt.extend_from_slice(images);
        let id = self.send_request(
            "session/prompt",
            json!({"sessionId": self.session_id, "prompt": prompt}),
        )?;
        self.prompt_in_flight = true;
        Ok(id)
    }

    /// Wait out an accepted prompt and fold it into a TurnResult.
    pub(crate) fn await_prompt(
        &mut self,
        id: u64,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<TurnResult, String> {
        let (result, unseen) = self.prompt_result(id, deadline, cancel)?;
        // `result.usage` wins when present and non-zero; a cancelled turn
        // carries none, so the last usage_update snapshot stands in.
        if let Some(u) = result.get("usage").and_then(usage_from) {
            self.usage = u;
        }
        let stop_reason = result
            .get("stopReason")
            .and_then(Value::as_str)
            .unwrap_or("");
        if stop_reason == "refusal" {
            return Err("Devin refused this request under its usage policy".to_string());
        }
        // A settled end_turn whose stream carried nothing — no answer
        // text, no thought, no tool call — is the silent usage-limit
        // failure mode: the prompt settled, so `prompt_in_flight` is
        // already clear and the caller pools the session for the host's
        // identical retry instead of folding an empty reply into
        // history. Keepalives ask for a near-empty reply by design and
        // never take this branch.
        if (stop_reason == "end_turn" || self.guard.cut().is_some())
            && !self.keepalive_prompt
            && self.text.trim().is_empty()
            && self.thought.trim().is_empty()
            && self.redirected.is_empty()
        {
            return Err(EMPTY_END_TURN.to_string());
        }
        let mut r = self.finish(stop_reason);
        r.unseen = unseen;
        Ok(r)
    }

    /// Record what this session just answered so the next request can be
    /// recognized as its strict continuation. An unseen answer (the relay
    /// client was already gone when the turn settled) records an empty
    /// echo: gray's history carries no assistant items for it, so the next
    /// request's echo zone is empty too.
    pub(crate) fn absorb(&mut self, turn: &PreparedTurn, r: &TurnResult) {
        self.system = turn.system.clone();
        self.absorbed = turn.input.clone();
        self.reply_call_ids = if r.unseen {
            Vec::new()
        } else {
            r.calls.iter().map(|(id, _, _)| id.clone()).collect()
        };
        self.reply_text = if r.unseen {
            String::new()
        } else {
            r.text.trim().to_string()
        };
        let now = Instant::now();
        self.last_used = now;
        self.last_prompt_at = Some(now);
    }

    /// Record a failed turn whose prompt still settled upstream: the
    /// input counts as absorbed (its text is already in the child's
    /// transcript) but no reply is on record, so a later request
    /// continues with only its own tail — the same shape as an unseen
    /// answer. Used for a failed first prompt on a fresh session, where
    /// keeping the empty pre-turn state would let an unrelated
    /// conversation claim a child already holding this one's transcript.
    /// The host's identical retry still misses by design — nothing
    /// extends `absorbed` — and respawns, exactly like today.
    pub(crate) fn absorb_failed(&mut self, turn: &PreparedTurn) {
        self.system = turn.system.clone();
        self.absorbed = turn.input.clone();
        self.reply_call_ids = Vec::new();
        self.reply_text = String::new();
        let now = Instant::now();
        self.last_used = now;
        self.last_prompt_at = Some(now);
    }

    /// Record a settled keepalive: bumps only the upstream-contact clock
    /// and the internal counter. `absorbed`, `reply_call_ids`,
    /// `reply_text` and `last_used` stay exactly as the last real turn
    /// left them — `take`'s strict-continuation check keys off gray's
    /// history, so the injected `.` turn (which lives only in the child's
    /// transcript) changes nothing about the next match.
    fn absorb_keepalive(&mut self) {
        self.keepalive_turns += 1;
        self.last_prompt_at = Some(Instant::now());
    }

    /// Drive one cache-refresh `session/prompt` while this session is out
    /// of the pool: fixed marker text, empty tool inventory (native calls
    /// are denied anyway), bounded by `KEEPALIVE_WINDOW`. The reply is
    /// discarded; its usage goes to the debug trace under `keepalive:N`,
    /// never into host accounting. Returns the upstream error on any
    /// failure — the caller then checks `prompt_in_flight` and only closes
    /// a session whose prompt never settled, the same rule as a failed
    /// real turn.
    fn keepalive(&mut self) -> Result<(), String> {
        let never = AtomicBool::new(false);
        let deadline = Instant::now() + KEEPALIVE_WINDOW;
        let next = self.keepalive_turns + 1;
        match self
            .send_prompt(KEEPALIVE_TEXT, &[], &[])
            .and_then(|id| self.await_prompt(id, deadline, &never))
        {
            Ok(r) => {
                self.absorb_keepalive();
                crate::chat::trace_turn(
                    &format!("keepalive:{next}"),
                    &self.session_id,
                    KEEPALIVE_TEXT.len(),
                    Some(&r),
                    None,
                    None,
                );
                Ok(())
            }
            Err(e) => {
                crate::chat::trace_turn(
                    &format!("keepalive:{next}"),
                    &self.session_id,
                    KEEPALIVE_TEXT.len(),
                    None,
                    Some(&e),
                    None,
                );
                Err(e)
            }
        }
    }

    /// Best-effort teardown: `session/delete` (≤5s), stdin closed, ≤2s
    /// grace, then kill.
    pub fn close(self) {
        self.close_inner(true);
    }

    /// End the child but keep the upstream session: its record stays (or
    /// was never written) for a later sidecar's [`reattach`].
    pub(crate) fn abandon(self) {
        self.close_inner(false);
    }

    fn close_inner(mut self, delete: bool) {
        if delete {
            unpersist(&self.session_id);
        }
        if delete
            && !self.session_id.is_empty()
            && self
                .send_request("session/delete", json!({"sessionId": self.session_id}))
                .is_ok()
        {
            let never = AtomicBool::new(false);
            let _ = self.recv_response(
                Duration::from_secs(5),
                Instant::now() + Duration::from_secs(5),
                &never,
            );
        }
        drop(self.stdin);
        let wait_deadline = Instant::now() + Duration::from_secs(2);
        while self.child.try_wait().ok().flatten().is_none() && Instant::now() < wait_deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn send(&mut self, v: &Value) -> Result<(), String> {
        let line = serde_json::to_string(v).map_err(|e| format!("frame encode: {e}"))? + "\n";
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.flush())
            .map_err(|_| "native stdin closed".to_string())
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<u64, String> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok(id)
    }

    fn answer_request(&mut self, id: &Value, result: Value) -> Result<(), String> {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    fn refuse_request(&mut self, id: &Value) -> Result<(), String> {
        self.send(&json!({"jsonrpc": "2.0", "id": id,
            "error": {"code": -32601, "message": "method not found"}}))
    }

    /// One server→client request: permissions always cancel, anything else
    /// is a JSON-RPC method-not-found (never a hang).
    fn on_request(&mut self, line: &Value) -> Result<(), String> {
        let id = line.get("id").cloned().unwrap_or(Value::Null);
        match line.get("method").and_then(Value::as_str) {
            Some("session/request_permission") if self.prompt_in_flight && !self.batch_flushed => {
                self.pending_perms.push(id);
                self.batch_quiet = Some(Instant::now() + BATCH_QUIET);
                Ok(())
            }
            Some("session/request_permission") => {
                self.answer_request(&id, json!({"outcome": {"outcome": "cancelled"}}))
            }
            _ => self.refuse_request(&id),
        }
    }

    fn on_notification(&mut self, line: &Value) {
        if self.loading || line.get("method").and_then(Value::as_str) != Some("session/update") {
            return;
        }
        let update = &line["params"]["update"];
        let kind = update.get("sessionUpdate").and_then(Value::as_str);
        if self.guard.cut().is_some() && kind != Some("usage_update") {
            return;
        }
        match kind {
            Some("agent_message_chunk") => {
                if let Some(t) = update.pointer("/content/text").and_then(Value::as_str) {
                    if self.text_boundary
                        && !self.text.is_empty()
                        && !self.text.ends_with(char::is_whitespace)
                        && !t.starts_with(char::is_whitespace)
                    {
                        self.text.push_str("\n\n");
                    }
                    self.text_boundary = false;
                    self.text.push_str(t);
                    if let Some(m) = self.guard.feed(&mut self.text) {
                        self.on_cut(m);
                    }
                }
            }
            Some("agent_thought_chunk") => {
                self.text_boundary = true;
                if let Some(t) = update.pointer("/content/text").and_then(Value::as_str) {
                    self.thought.push_str(t);
                }
            }
            Some("tool_call") => {
                self.text_boundary = true;
                // The model reached for native tools instead of the text
                // funnel. Every call of the step is collected (they arrive
                // together, each parked on a held permission request) and
                // the turn is cancelled once the batch goes quiet.
                // execute/read/web calls redirect onto real tools;
                // anything else lands on a self-explaining bash echo — a
                // normal result, not a "does not exist" error card.
                let id = update
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !self.batch_flushed && self.seen_calls.insert(id) {
                    let call = match redirect_call(update, &self.names) {
                        Some(call) => call,
                        None => blocked_call(update, &self.names),
                    };
                    self.redirected.push(call);
                    self.batch_quiet = Some(Instant::now() + BATCH_QUIET);
                }
            }
            Some("tool_call_update") => {
                self.text_boundary = true;
                let id = update
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !self.redirected.is_empty()
                    && !self.seen_calls.contains(id)
                    && update.pointer("/_meta/cognition.ai~1canceled") == Some(&json!(true))
                {
                    self.dropped_siblings += 1;
                }
            }
            Some("usage_update") => {
                self.text_boundary = true;
                // Keep the LAST snapshot: a later update carries the final
                // counters for this prompt.
                if let Some(u) = usage_from(update) {
                    self.usage = u;
                }
            }
            _ => {
                self.text_boundary = true;
            }
        }
    }

    fn on_cut(&mut self, marker: &str) {
        let at = self.text.len();
        crate::chat::pool_event(
            &format!("turn_cut:{}\tat={at}", marker_name(marker)),
            &self.session_id,
        );
        eprintln!(
            "devin-sub: reply cut at fabricated {} record (offset {at}); upstream cancelled",
            marker_name(marker)
        );
        if !self.keepalive_prompt {
            self.prior_cut = true;
        }
        if self.redirected.is_empty() {
            let _ = self.send(&json!({"jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": self.session_id}}));
        }
    }

    fn flush_batch(&mut self) {
        self.batch_quiet = None;
        self.batch_flushed = true;
        let _ = self.send(&json!({"jsonrpc": "2.0",
            "method": "session/cancel",
            "params": {"sessionId": self.session_id}}));
        for id in std::mem::take(&mut self.pending_perms) {
            let _ = self.answer_request(&id, json!({"outcome": {"outcome": "cancelled"}}));
        }
        if self.redirected.len() > 1 {
            crate::chat::pool_event(
                &format!("batch_native\tcalls={}", self.redirected.len()),
                &self.session_id,
            );
        }
    }

    pub(crate) fn note_cut(&mut self, delta: String) -> String {
        let delta = match std::mem::take(&mut self.prior_dropped) {
            0 => delta,
            n => format!("{}\n\n{delta}", batch_cut_note(n)),
        };
        if std::mem::take(&mut self.prior_cut) {
            format!("{CUT_NOTE}\n\n{delta}")
        } else {
            delta
        }
    }

    /// Wait for the next response on the wire, dispatching notifications
    /// and server requests as they arrive. Wakes at least every
    /// `CANCEL_POLL` to check `cancel`: a relayed client disconnect must
    /// end the wait, never hang it.
    fn recv_response(
        &mut self,
        window: Duration,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Value, String> {
        let until = (Instant::now() + window).min(deadline);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(CLIENT_GONE.to_string());
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("Devin request timed out".into());
            }
            let wait = self.batch_quiet.map_or(CANCEL_POLL, |q| {
                q.saturating_duration_since(Instant::now()).min(CANCEL_POLL)
            });
            match self.rx.recv_timeout(left.min(wait)) {
                Ok(line) => {
                    if line.get("method").is_some() && line.get("id").is_some() {
                        self.on_request(&line)?;
                    } else if line.get("method").is_some() {
                        self.on_notification(&line);
                    } else if line.get("id").is_some() {
                        return Ok(line);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("native stdout closed".into());
                }
            }
            if self.batch_quiet.is_some_and(|q| Instant::now() >= q) {
                self.flush_batch();
            }
        }
    }

    /// Read until the response to request `id` arrives (or a hard error).
    fn call(
        &mut self,
        method: &str,
        params: Value,
        window: Duration,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Value, String> {
        let id = self.send_request(method, params)?;
        loop {
            let line = self.recv_response(window, deadline, cancel)?;
            if line.get("id") == Some(&json!(id)) {
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, &self.stderr_tail));
                }
                return Ok(line.get("result").cloned().unwrap_or(Value::Null));
            }
            // A response to another in-flight request: we send one at a
            // time, so anything else is protocol noise — keep waiting.
        }
    }

    /// Drain notifications until the `session/prompt` result arrives,
    /// plus an `unseen` flag. When the wait fails the prompt may still be
    /// live upstream, so `session/cancel` goes out once and the wire gets
    /// ≤10s for the result to settle. A settled result returns — `unseen`
    /// only when the client went away (the reply never reaches it and
    /// the session records an empty echo; any other settle is still
    /// deliverable) — and a settled error returns that error; either way
    /// `prompt_in_flight` clears, leaving a known session state the
    /// caller can pool. Only a prompt that won't settle keeps the flag
    /// and costs the session.
    fn prompt_result(
        &mut self,
        id: u64,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(Value, bool), String> {
        loop {
            let line = match self.recv_response(Duration::from_secs(600), deadline, cancel) {
                Ok(line) => line,
                Err(e) => {
                    let _ = self.send(&json!({"jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": self.session_id}}));
                    let grace = Instant::now() + Duration::from_secs(10);
                    let never = AtomicBool::new(false);
                    while Instant::now() < grace {
                        match self.recv_response(Duration::from_secs(10), grace, &never) {
                            Ok(line) if line.get("id") == Some(&json!(id)) => {
                                self.prompt_in_flight = false;
                                self.last_prompt_at = Some(Instant::now());
                                if let Some(err) = line.get("error") {
                                    return Err(map_rpc_error(err, &self.stderr_tail));
                                }
                                return Ok((
                                    line.get("result").cloned().unwrap_or(Value::Null),
                                    e == CLIENT_GONE,
                                ));
                            }
                            Ok(_) => {}
                            Err(_) => break,
                        }
                    }
                    return Err(e);
                }
            };
            if line.get("id") == Some(&json!(id)) {
                self.prompt_in_flight = false;
                self.last_prompt_at = Some(Instant::now());
                if let Some(err) = line.get("error") {
                    return Err(map_rpc_error(err, &self.stderr_tail));
                }
                return Ok((line.get("result").cloned().unwrap_or(Value::Null), false));
            }
        }
    }

    /// Fold the accumulated reply into a TurnResult.
    fn finish(&mut self, stop_reason: &str) -> TurnResult {
        let turn_id = rand_hex(8);
        if self.dropped_siblings > 0 && !self.keepalive_prompt {
            self.prior_dropped = self.dropped_siblings;
            crate::chat::pool_event(
                &format!("batch_cut\tdropped={}", self.dropped_siblings),
                &self.session_id,
            );
        }
        // Redirected native calls never mix with funnel calls: redirect wins.
        let salvage = self.guard.salvage();
        let mut text = self.text.trim().to_string();
        let calls: Vec<(String, String)> = if !self.redirected.is_empty() {
            if let Some(at) = salvage {
                text = self.text[..at].trim().to_string();
            }
            self.redirected.clone()
        } else {
            let head = salvage.map_or(self.text.as_str(), |at| &self.text[..at]);
            match parse_calls_block(head, &self.names) {
                Some((before, calls)) => {
                    text = before.trim().to_string();
                    calls
                }
                None => match salvage {
                    Some(at) => {
                        let calls = salvage_calls(&self.text[at..], &self.names);
                        text = self.text[..at].trim().to_string();
                        if calls.is_empty() {
                            crate::chat::pool_event(
                                &format!("turn_cut:Tool_call\tat={at}"),
                                &self.session_id,
                            );
                            eprintln!(
                                "devin-sub: reply cut at fabricated Tool_call record (offset {at}); nothing salvageable"
                            );
                        } else {
                            crate::chat::pool_event(
                                &format!("turn_salvage\tcalls={}", calls.len()),
                                &self.session_id,
                            );
                            eprintln!(
                                "devin-sub: salvaged {} fabricated Tool_call record(s) (offset {at})",
                                calls.len()
                            );
                        }
                        calls
                    }
                    None => Vec::new(),
                },
            }
        };
        let text = strip_reasoning(&text).trim().to_string();
        let calls: Vec<(String, String, String)> = calls
            .into_iter()
            .enumerate()
            .map(|(i, (name, args))| (format!("call_{turn_id}_{}", i + 1), name, unstage(args)))
            .collect();
        let stop = match stop_reason {
            "max_tokens" => "incomplete:max_output_tokens".to_string(),
            _ if !calls.is_empty() => "tool_use".to_string(),
            _ => "completed".to_string(),
        };
        TurnResult {
            text,
            thought: self.thought.trim().to_string(),
            calls,
            usage: self.usage.clone(),
            stop,
            unseen: false,
        }
    }
}

fn unstage(args: String) -> String {
    let Ok(cwd) = std::env::current_dir() else {
        return args;
    };
    let cwd = cwd.to_string_lossy();
    STAGE
        .iter()
        .filter_map(OnceLock::get)
        .fold(args, |a, s| a.replace(&*s.work.to_string_lossy(), &cwd))
}

/// Map a JSON-RPC/CLI failure onto the host-facing error classes.
fn map_rpc_error(err: &Value, stderr: &Arc<Mutex<VecDeque<String>>>) -> String {
    let mut detail = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if detail.trim().is_empty()
        && let Ok(tail) = stderr.lock()
        && let Some(last) = tail.iter().rev().find(|line| !line.trim().is_empty())
    {
        detail = last.clone();
    }
    let detail = redact(&detail);
    let low = detail.to_lowercase();
    if low.contains("quota") || low.contains("reached free model rate limit") {
        return format!("Devin usage limit reached: {detail}");
    }
    if low.contains("429") || low.contains("rate limit") {
        return format!("Devin rate limited: {detail}");
    }
    if low.contains("not logged in") || low.contains("authenticate") || low.contains("unauthorized")
    {
        return setup::LOGIN_HINT.to_string();
    }
    if detail.trim().is_empty() {
        "native request failed".to_string()
    } else {
        format!("native request failed: {detail}")
    }
}

/// Redact token|secret|key|password-looking values from a detail string.
fn redact(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for word in s.split_whitespace() {
        let low = word.to_lowercase();
        let secretish = ["token", "secret", "key", "password"]
            .iter()
            .any(|k| low.contains(k) && (low.contains('=') || low.ends_with(':')));
        if secretish {
            let mut it = word.splitn(2, ['=', ':']);
            out.push_str(it.next().unwrap_or(word));
            out.push_str(if word.contains('=') {
                "=<redacted>"
            } else {
                ": <redacted>"
            });
        } else {
            out.push_str(word);
        }
        out.push(' ');
    }
    out.trim_end().to_string()
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
