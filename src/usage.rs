//! `provider/usage`: Devin subscription quota from the CLI's cached
//! `GetUserStatus` payload.
//!
//! `devin auth status` refreshes `~/.cache/devin/cli/user_status.<id>.bin`
//! (JSON envelope: `{version, identity_digest, fetched_at_secs, payload}`
//! — payload is the base64'd protobuf response). The plan block inside
//! carries the plan name, the monthly billing window and the Devin ACU
//! limits the TUI's quota line draws from. A stale cache (older than
//! [`REFRESH_AFTER`]) is refreshed by spawning `devin auth status`
//! first, so `/usage` data tracks the real fetch cadence.
//!
//! Field numbers were recovered by decoding the cache file's wire format
//! against the binary's proto field names (`monthly_prompt_credits`,
//! `devin_info`, `hide_weekly_quota`, `top_up_*`, …). Proto3 defaults are
//! omitted, so a missing field is "unset", not zero.

use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

use base64::Engine;
use prost::Message;
use serde_json::{Value, json};

use gray_plugin::ProviderRpcError;

/// The CLI cache is reused for this long; older files get an
/// `auth status` refresh first (~3s online).
const REFRESH_AFTER: Duration = Duration::from_secs(30 * 60);
/// Cap on the refresh command itself.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(20);

/// The CLI's `user_status` cache dir. With an account home the child env
/// pins `XDG_CACHE_HOME` to `<home>/cache`, so the cache — and this read
/// — follows it: quota windows can't bleed across accounts.
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
fn latest_cache_for(home: Option<&std::path::Path>) -> Option<(PathBuf, u64)> {
    let dir = cache_dir_for(home)?;
    let mut best: Option<(PathBuf, u64)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !(name.starts_with("user_status.") && name.ends_with(".bin")) {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
            best = Some((entry.path(), mtime));
        }
    }
    best
}

/// `provider/usage` entry point.
pub fn handle() -> Result<Value, ProviderRpcError> {
    handle_in(None)
}

/// [`handle`] against an account's credential home — its
/// `XDG_CACHE_HOME` (`<home>/cache`) decides which `user_status` cache
/// this reads and which login the refresh probes.
pub fn handle_in(home: Option<&std::path::Path>) -> Result<Value, ProviderRpcError> {
    let Some((_, mtime)) = latest_cache_for(home) else {
        return Err(ProviderRpcError::Unavailable(
            "no Devin CLI user-status cache; run `devin auth status` once".into(),
        ));
    };
    if crate::setup::unix_secs().saturating_sub(mtime) > REFRESH_AFTER.as_secs() {
        refresh(home);
    }
    let Some((path, _)) = latest_cache_for(home) else {
        return Err(ProviderRpcError::Unavailable(
            "user-status cache vanished".into(),
        ));
    };
    let payload = read_payload(&path)
        .ok_or_else(|| ProviderRpcError::Unavailable("unreadable user-status cache".into()))?;
    let status = Status::decode(&*payload)
        .map_err(|_| ProviderRpcError::Unavailable("undecodable user-status payload".into()))?;
    Ok(map_usage(&status))
}

/// `devin auth status` re-fetches user status when online; a hard
/// timeout keeps a hung refresh from stalling `/usage`.
fn refresh(home: Option<&std::path::Path>) {
    if let Some(binary) = crate::setup::resolve_command() {
        let _ = crate::setup::run_captured_in(
            &binary,
            &["auth", "status"],
            REFRESH_TIMEOUT,
            &crate::setup::child_env_for(home),
        );
    }
}

/// `{version, identity_digest, fetched_at_secs, payload}` — payload is
/// base64 protobuf.
fn read_payload(path: &std::path::Path) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let b64 = v.get("payload").and_then(Value::as_str)?;
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

// ---- recovered GetUserStatus layout (unknown fields skipped) ---------

/// Walk a protobuf message by field-number path, returning the raw field
/// bytes for length-delimited fields or the varint for varint fields.
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
