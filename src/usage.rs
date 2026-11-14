//! `provider/usage`: Devin subscription quota from the `GetUserStatus`
//! Connect RPC the CLI itself calls.
//!
//! The CLI used to leave that answer in `~/.cache/devin/cli/user_status.*.bin`;
//! it no longer writes the file at all — `devin auth status` fetches, prints
//! identity/plan text and drops the quota. So this module asks the same
//! service itself: `POST <api_server_url>/exa.seat_management_pb
//! .SeatManagementService/GetUserStatus` with `content-type:
//! application/proto`, `connect-protocol-version: 1`, the stored key as
//! Basic auth and again inside the request's client metadata (the server
//! authenticates on the metadata field), decoded with the proto below.
//! Credentials come from `<data>/devin/credentials.toml` — the same file
//! (`windsurf_api_key`, `api_server_url`) `devin` reads for that home.
//!
//! The plan block inside `user_status` carries the plan name, the monthly
//! billing window and the Devin ACU limits the TUI's quota line draws
//! from. CLIs old enough to still write the `user_status` cache are read
//! from it when the call fails. Error text stays clear of the words
//! Gray's picker reads as a dead login (`401`, `unauthorized`, `login`,
//! `auth`, …) unless the login is actually gone.
//!
//! Field numbers were recovered by decoding the wire format against the
//! binary's proto field names (`monthly_prompt_credits`, `devin_info`,
//! `hide_weekly_quota`, `top_up_*`, …). Proto3 defaults are omitted, so a
//! missing field is "unset", not zero.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use base64::Engine;
use prost::Message;
use serde_json::{Value, json};

use gray_plugin::ProviderRpcError;

/// Cap on the usage round trip (~0.5s against a live API).
const RPC_TIMEOUT: Duration = Duration::from_secs(15);
/// Cap on the `devin --version` lookup that names this client.
const VERSION_TIMEOUT: Duration = Duration::from_secs(5);
const RPC_PATH: &str = "/exa.seat_management_pb.SeatManagementService/GetUserStatus";
const DEFAULT_SERVER: &str = "https://server.codeium.com";
/// What the CLI itself sends before it knows its version.
const FALLBACK_CLIENT_VERSION: &str = "0.0.0-dev";

/// The CLI's data dir for this home: an account home replaces
/// `XDG_DATA_HOME`, so its `devin/credentials.toml` is the one read here.
fn data_dir(home: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Some(h) = home {
        return Some(h.to_path_buf());
    }
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(home.join(".local").join("share"))
}

struct Credentials {
    api_key: String,
    server: String,
}

fn toml_string(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (k, v) = line.split_once('=')?;
        (k.trim() == key)
            .then(|| v.trim().trim_matches('"').to_string())
            .filter(|v| !v.is_empty())
    })
}

fn credentials(home: Option<&std::path::Path>) -> Option<Credentials> {
    let text =
        std::fs::read_to_string(data_dir(home)?.join("devin").join("credentials.toml")).ok()?;
    Some(Credentials {
        api_key: toml_string(&text, "windsurf_api_key")?,
        server: toml_string(&text, "api_server_url").unwrap_or_else(|| DEFAULT_SERVER.to_string()),
    })
}

/// Version the API gates the reply on (`devin 3000.11.234 (hash)`).
fn client_version() -> String {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            crate::setup::resolve_command()
                .and_then(|binary| {
                    crate::setup::run_captured(&binary, &["--version"], VERSION_TIMEOUT).ok()
                })
                .and_then(|(ok, out, _)| {
                    if !ok {
                        return None;
                    }
                    out.split_whitespace().nth(1).map(str::to_string)
                })
                .unwrap_or_else(|| FALLBACK_CLIENT_VERSION.to_string())
        })
        .clone()
}

/// `provider/usage` entry point.
pub fn handle() -> Result<Value, ProviderRpcError> {
    handle_in(None)
}

/// [`handle`] against an account's credential home — the same home the
/// CLI's turns run in, so the quota follows the account that serves them.
pub fn handle_in(home: Option<&std::path::Path>) -> Result<Value, ProviderRpcError> {
    match fetch_user_status(home) {
        Ok(status) => Ok(map_usage(&status)),
        Err(rpc) => match cached_user_status(home) {
            Some(status) => Ok(map_usage(&status)),
            None => Err(rpc),
        },
    }
}

fn fetch_user_status(home: Option<&std::path::Path>) -> Result<Status, ProviderRpcError> {
    let Some(cred) = credentials(home) else {
        return Err(ProviderRpcError::Unavailable(
            "Devin CLI is logged out in this home; run `devin auth login` once".into(),
        ));
    };
    let version = client_version();
    let request = Request {
        metadata: Some(Metadata {
            ide_name: Some("chisel".into()),
            extension_version: Some(version.clone()),
            api_key: Some(cred.api_key.clone()),
            locale: Some("en".into()),
            os: Some(std::env::consts::OS.into()),
            ide_version: Some(version),
            extension_name: Some("chisel".into()),
        }),
    };
    let url = format!("{}{RPC_PATH}", cred.server.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(RPC_TIMEOUT)
        .build()
        .map_err(|e| ProviderRpcError::Unavailable(format!("Devin usage client failed: {e}")))?;
    let response = client
        .post(&url)
        .header("content-type", "application/proto")
        .header("connect-protocol-version", "1")
        .header("authorization", format!("Basic {}", cred.api_key))
        .body(request.encode_to_vec())
        .send()
        .map_err(|e| {
            ProviderRpcError::Unavailable(format!("could not reach the Devin API: {e}"))
        })?;
    let code = response.status().as_u16();
    let body = response
        .bytes()
        .map_err(|e| ProviderRpcError::Unavailable(format!("Devin usage reply cut short: {e}")))?;
    if !(200..300).contains(&code) {
        return Err(rpc_failure(home, code));
    }
    let decoded = Response::decode(&*body)
        .map_err(|_| ProviderRpcError::Unavailable("undecodable Devin usage reply".into()))?;
    decoded.user_status.ok_or_else(|| {
        ProviderRpcError::Unavailable("Devin usage reply carried no user status".into())
    })
}

/// A failed call is only a dead login when the API says so or the CLI
/// itself reports one — otherwise it is an ordinary fetch failure.
fn rpc_failure(home: Option<&std::path::Path>, code: u16) -> ProviderRpcError {
    let logged_out = crate::setup::probe_login_in(home) == crate::setup::LoginState::LoggedOut;
    if code == 401 || code == 403 || logged_out {
        return ProviderRpcError::Unavailable(format!(
            "Devin rejected this login (HTTP {code}); run `devin auth login` once"
        ));
    }
    ProviderRpcError::Unavailable(format!("Devin usage request failed (HTTP {code})"))
}

/// The CLI's `user_status` cache dir, still written by older builds. With
/// an account home the child env pins `XDG_CACHE_HOME` to `<home>/cache`,
/// so the cache — and this read — follows it.
fn cache_dir_for(home: Option<&std::path::Path>) -> Option<PathBuf> {
    if let Some(h) = home {
        return Some(h.join("cache").join("devin/cli"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(home.join(".cache/devin/cli"))
}

/// The newest `user_status.*.bin`, if any.
fn latest_cache_for(home: Option<&std::path::Path>) -> Option<PathBuf> {
    std::fs::read_dir(cache_dir_for(home)?)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if !(name.starts_with("user_status.") && name.ends_with(".bin")) {
                return None;
            }
            let mtime = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some((mtime, entry.path()))
        })
        .max_by_key(|(mtime, _)| *mtime)
        .map(|(_, path)| path)
}

fn cached_user_status(home: Option<&std::path::Path>) -> Option<Status> {
    let path = latest_cache_for(home)?;
    let payload = read_payload(&path)?;
    Status::decode(&*payload).ok()
}

/// `{version, identity_digest, fetched_at_secs, payload}` — payload is
/// base64 protobuf.
fn read_payload(path: &std::path::Path) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let b64 = v.get("payload").and_then(Value::as_str)?;
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

#[derive(Clone, PartialEq, Message)]
struct Status {
    /// Plan block: windows live directly on it, names under `.features`.
    #[prost(message, tag = "13")]
    plan: Option<Plan>,
}

#[derive(Clone, PartialEq, Message)]
struct Plan {
    #[prost(message, tag = "1")]
    features: Option<Features>,
    #[prost(uint64, optional, tag = "14")]
    daily_remaining: Option<u64>,
    #[prost(uint64, optional, tag = "15")]
    weekly_remaining: Option<u64>,
    #[prost(uint64, optional, tag = "17")]
    daily_reset: Option<u64>,
    #[prost(uint64, optional, tag = "18")]
    weekly_reset: Option<u64>,
}

#[derive(Clone, PartialEq, Message)]
struct Features {
    #[prost(string, optional, tag = "2")]
    plan_name: Option<String>,
    #[prost(message, tag = "33")]
    devin_info: Option<DevinInfo>,
}

#[derive(Clone, PartialEq, Message)]
struct DevinInfo {
    #[prost(message, tag = "12")]
    hint: Option<Hint>,
}

#[derive(Clone, PartialEq, Message)]
struct Hint {
    #[prost(string, optional, tag = "2")]
    text: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct Request {
    #[prost(message, optional, tag = "1")]
    metadata: Option<Metadata>,
}

#[derive(Clone, PartialEq, Message)]
struct Metadata {
    #[prost(string, optional, tag = "1")]
    ide_name: Option<String>,
    #[prost(string, optional, tag = "2")]
    extension_version: Option<String>,
    #[prost(string, optional, tag = "3")]
    api_key: Option<String>,
    #[prost(string, optional, tag = "4")]
    locale: Option<String>,
    #[prost(string, optional, tag = "5")]
    os: Option<String>,
    #[prost(string, optional, tag = "7")]
    ide_version: Option<String>,
    #[prost(string, optional, tag = "12")]
    extension_name: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct Response {
    /// `GetUserStatusResponse.user_status` — the message the old cache
    /// stored whole, so `Status` decodes it unchanged.
    #[prost(message, optional, tag = "1")]
    user_status: Option<Status>,
}

fn iso(secs: u64) -> String {
    chrono::DateTime::from_timestamp(secs as i64, 0)
        .map(|t| t.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_default()
}

/// Map the decoded `GetUserStatusResponse` into usage windows.
///
/// Recovered layout (verified against the cached payload, field numbers
/// in the binary's `PlanStatus` declaration order):
/// - `13` plan block: `.1` = plan/team features msg, `.2`/`.3` = monthly
///   period start/end timestamps, `.14` = daily_quota_remaining_percent,
///   `.15` = weekly_quota_remaining_percent, `.16` = overage_balance_micros,
///   `.17` = daily_quota_reset_at_unix, `.18` = weekly_quota_reset_at_unix
///   (`.19`/`.20` acu_consumed/acu_limit are absent on unmetered plans)
/// - `13.1` = plan features: `.2` plan_name, `.7` monthly flow credits,
///   `.8` monthly prompt credits, `.24` team features, `.33` devin_info
fn map_usage(status: &Status) -> Value {
    let mut windows = Vec::new();
    let mut notes = Vec::new();

    let plan = status.plan.as_ref();
    let plan_name = plan
        .and_then(|p| p.features.as_ref())
        .and_then(|f| f.plan_name.clone());
    // The two quota windows the TUI's own quota line draws from: the
    // server publishes what is *left* (percent) plus the reset instant;
    // `used_percent` is the complement. A percent above 100 is an unset
    // sentinel, not a window. Monthly plan ceilings stay out: they aren't
    // the quota this card is for.
    let quota = |remaining: u64, reset: u64, id: &str, label: &str, kind: &str, mins: u64| {
        (remaining <= 100).then(|| {
            let mut w = json!({
                "id": id,
                "label": label,
                "kind": kind,
                "duration_mins": mins,
                "used_percent": (100 - remaining) as f64,
            });
            if reset > 0 {
                w["resets_at"] = json!(iso(reset));
            }
            w
        })
    };
    if let Some(left) = plan.and_then(|p| p.daily_remaining) {
        windows.extend(quota(
            left,
            plan.and_then(|p| p.daily_reset).unwrap_or(0),
            "daily",
            "Daily",
            "session",
            24 * 60,
        ));
    }
    if let Some(left) = plan.and_then(|p| p.weekly_remaining) {
        windows.extend(quota(
            left,
            plan.and_then(|p| p.weekly_reset).unwrap_or(0),
            "weekly",
            "Weekly",
            "weekly",
            7 * 24 * 60,
        ));
    }
    if let Some(hint) = plan
        .and_then(|p| p.features.as_ref())
        .and_then(|f| f.devin_info.as_ref())
        .and_then(|d| d.hint.as_ref())
        .and_then(|h| h.text.clone())
    {
        notes.push(hint);
    }

    json!({
        "available": true,
        "title": "Devin",
        "plan": plan_name.unwrap_or_else(|| "unknown".into()),
        "windows": windows,
        "checked_at": iso(crate::setup::unix_secs()),
        "note": if notes.is_empty() { Value::Null } else { json!(notes.join(" · ")) },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        while n > 0x7f {
            out.push((n as u8 & 0x7f) | 0x80);
            n >>= 7;
        }
        out.push(n as u8);
        out
    }
    fn field(num: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(num << 3 | 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }
    fn vfield(num: u64, v: u64) -> Vec<u8> {
        let mut out = varint(num << 3);
        out.extend(varint(v));
        out
    }

    /// Synthetic GetUserStatusResponse with the recovered layout:
    /// plan block 13 { 1 { 2 plan_name, 7 flow credits, 8 prompt credits,
    /// 33 { 12 { 2 hint } } }, 2/3 period stamps, 14/15 daily/weekly
    /// remaining percent, 17/18 their reset instants }.
    #[test]
    fn maps_plan_block() {
        let devin_info = field(12, &field(2, b"Ask your account admin to raise it"));
        let mut features = field(2, b"Max");
        features.extend(vfield(7, 16384));
        features.extend(vfield(8, 600));
        features.extend(field(33, &devin_info));
        let mut plan = field(1, &features);
        plan.extend(field(2, &vfield(1, 1791122762)));
        plan.extend(field(3, &vfield(1, 1793801162)));
        plan.extend(vfield(14, 100));
        plan.extend(vfield(15, 74));
        plan.extend(vfield(17, 1791792000));
        plan.extend(vfield(18, 1792310400));
        let mut payload = vfield(1, 1);
        payload.extend(field(13, &plan));

        let status = Status::decode(&*payload).expect("wire bytes must decode");
        let limits = map_usage(&status);
        assert_eq!(limits["plan"], "Max");
        let windows = limits["windows"].as_array().unwrap();
        let ids: Vec<&str> = windows.iter().map(|w| w["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["daily", "weekly"], "{ids:?}");
        let daily = &windows[0];
        assert_eq!(daily["used_percent"], 0.0);
        assert_eq!(daily["resets_at"], "2026-10-12T08:00:00Z");
        let weekly = &windows[1];
        assert_eq!(weekly["used_percent"], 26.0);
        assert_eq!(weekly["kind"], "weekly");
        assert_eq!(weekly["resets_at"], "2026-10-18T08:00:00Z");
        let note = limits["note"].as_str().unwrap();
        assert!(!note.contains("billing period"), "{note}");
        assert!(note.contains("Ask your account admin"), "{note}");
    }

    #[test]
    fn empty_payload_still_available() {
        let limits = map_usage(&Status::default());
        assert_eq!(limits["available"], true);
        assert_eq!(limits["windows"].as_array().unwrap().len(), 0);
    }
}
