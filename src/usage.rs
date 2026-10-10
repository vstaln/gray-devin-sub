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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use gray_plugin::ProviderRpcError;

/// The CLI cache is reused for this long; older files get an
/// `auth status` refresh first (~3s online).
const REFRESH_AFTER: Duration = Duration::from_secs(30 * 60);
/// Cap on the refresh command itself.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(20);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    Some(home.join(".cache/devin/cli"))
}

/// The newest `user_status.*.bin`, if any.
fn latest_cache() -> Option<(PathBuf, u64)> {
    let dir = cache_dir()?;
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
    let Some((_, mtime)) = latest_cache() else {
        return Err(ProviderRpcError::Unavailable(
            "no Devin CLI user-status cache; run `devin auth status` once".into(),
        ));
    };
    if now_secs().saturating_sub(mtime) > REFRESH_AFTER.as_secs() {
        refresh();
    }
    let Some((path, _)) = latest_cache() else {
        return Err(ProviderRpcError::Unavailable(
            "user-status cache vanished".into(),
        ));
    };
    let payload = read_payload(&path)
        .ok_or_else(|| ProviderRpcError::Unavailable("unreadable user-status cache".into()))?;
    let limits = map_usage(&payload);
    Ok(limits)
}

/// `devin auth status` re-fetches user status when online; the spawned
/// child gets a hard timeout so a hung refresh can't stall `/usage`.
fn refresh() {
    let mut child = match std::process::Command::new("devin")
        .args(["auth", "status"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return,
    };
    let deadline = std::time::Instant::now() + REFRESH_TIMEOUT;
    while std::time::Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// `{version, identity_digest, fetched_at_secs, payload}` — payload is
/// base64 protobuf.
fn read_payload(path: &std::path::Path) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let b64 = v.get("payload").and_then(Value::as_str)?;
    b64_decode(b64)
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut n = 0;
    for c in s.bytes() {
        let Some(v) = T.iter().position(|&t| t == c) else {
            if c == b'=' {
                break;
            }
            continue;
        };
        acc = (acc << 6) | v as u32;
        n += 1;
        if n == 4 {
            out.extend_from_slice(&acc.to_be_bytes()[1..]);
            acc = 0;
            n = 0;
        }
    }
    // Tail group: 3 sextets = 18 bits → top 2 bytes; 2 sextets = 12 bits
    // → top 1 byte. The sextets sit in acc's low bits, so shift the
    // partial-byte padding off before slicing.
    if n == 3 {
        out.extend_from_slice(&(acc >> 2).to_be_bytes()[2..]);
    } else if n == 2 {
        out.push((acc >> 4) as u8);
    }
    Some(out)
}

// ---- minimal proto wire reader -----------------------------------------

/// Walk a protobuf message by field-number path, returning the raw field
/// bytes for length-delimited fields or the varint for varint fields.
#[derive(Clone, Copy)]
enum Field<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

fn each_field<'a>(b: &'a [u8], mut f: impl FnMut(u64, Field<'a>)) {
    fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
        let mut n = 0u64;
        let mut s = 0;
        loop {
            let byte = *b.get(*i)?;
            n |= ((byte & 0x7f) as u64) << s;
            s += 7;
            *i += 1;
            if byte & 0x80 == 0 {
                return Some(n);
            }
            if s > 63 {
                return None;
            }
        }
    }
    let mut i = 0usize;
    while i < b.len() {
        let Some(tag) = varint(b, &mut i) else { break };
        let (num, wt) = (tag >> 3, tag & 7);
        match wt {
            0 => {
                let Some(v) = varint(b, &mut i) else { break };
                f(num, Field::Varint(v));
            }
            5 => {
                let Some(v) = b.get(i..i + 4) else { break };
                f(num, Field::Bytes(v));
                i += 4;
            }
            1 => {
                let Some(v) = b.get(i..i + 8) else { break };
                f(num, Field::Bytes(v));
                i += 8;
            }
            2 => {
                let Some(len) = varint(b, &mut i) else { break };
                let Some(v) = b.get(i..i + len as usize) else {
                    break;
                };
                f(num, Field::Bytes(v));
                i += len as usize;
            }
            _ => break,
        }
    }
}

fn varint_at(b: &[u8], path: &[u64]) -> Option<u64> {
    let (&head, rest) = path.split_first()?;
    let mut hit = None;
    each_field(b, |num, f| {
        if num != head || hit.is_some() {
            return;
        }
        match (rest.is_empty(), f) {
            (true, Field::Varint(v)) => hit = Some(v),
            (false, Field::Bytes(inner)) => hit = varint_at(inner, rest),
            _ => {}
        }
    });
    hit
}

fn bytes_at<'a>(b: &'a [u8], path: &[u64]) -> Option<&'a [u8]> {
    let (&head, rest) = path.split_first()?;
    let mut hit = None;
    each_field(b, |num, f| {
        if num != head || hit.is_some() {
            return;
        }
        match (rest.is_empty(), f) {
            (true, Field::Bytes(v)) => hit = Some(v),
            (false, Field::Bytes(inner)) => hit = bytes_at(inner, rest),
            _ => {}
        }
    });
    hit
}

fn str_at(b: &[u8], path: &[u64]) -> Option<String> {
    let raw = bytes_at(b, path)?;
    std::str::from_utf8(raw).ok().map(str::to_string)
}

fn iso(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let tod = secs % 86400;
    let (y, m, d) = civil(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        tod % 3600 / 60,
        tod % 60
    )
}

fn civil(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Map the decoded `GetUserStatusResponse` into usage windows.
///
/// Recovered layout (verified against the cached payload):
/// - `13` plan block: `.1` = plan/team features msg, `.2`/`.3` = monthly
///   period start/end timestamps, `.14` = weekly ACU limit, `.17`/`.18` =
///   daily-window bounds, `.33` = devin_info
/// - `13.1` = plan features: `.2` plan_name, `.7` monthly flow credits,
///   `.8` monthly prompt credits, `.24` team features, `.33` devin_info
fn map_usage(payload: &[u8]) -> Value {
    let mut windows = Vec::new();
    let mut notes = Vec::new();

    let plan_name = str_at(payload, &[13, 1, 2]);
    // Weekly ACU limit (field 14) — Devin metered models draw from a
    // weekly bucket; swe-2 family is unmetered. No used counter is
    // published (enforcement is server-side), so the row renders as a
    // 0→limit scale. Monthly plan ceilings stay out: they aren't the
    // quota this card is for.
    if let Some(limit) = varint_at(payload, &[13, 14]).filter(|v| *v > 0) {
        windows.push(json!({
            "id": "weekly_acu",
            "label": "Weekly ACU",
            "kind": "weekly",
            "limit": limit,
            "unit": "ACU",
        }));
    }
    if let Some(hint) = str_at(payload, &[13, 1, 33, 12, 2]) {
        notes.push(hint);
    }

    json!({
        "available": true,
        "title": "Devin",
        "plan": plan_name.unwrap_or_else(|| "unknown".into()),
        "windows": windows,
        "checked_at": iso(now_secs()),
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
    /// 33 { 12 { 2 hint } } }, 2/3 period stamps, 14 weekly ACU }.
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
        let mut payload = vfield(1, 1);
        payload.extend(field(13, &plan));

        let limits = map_usage(&payload);
        assert_eq!(limits["plan"], "Max");
        let windows = limits["windows"].as_array().unwrap();
        let ids: Vec<&str> = windows.iter().map(|w| w["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"weekly_acu"), "{ids:?}");
        assert!(!ids.contains(&"plan_field_7"), "{ids:?}");
        assert!(!ids.contains(&"plan_field_8"), "{ids:?}");
        let acu = windows.iter().find(|w| w["id"] == "weekly_acu").unwrap();
        assert_eq!(acu["limit"], 100);
        let note = limits["note"].as_str().unwrap();
        assert!(!note.contains("billing period"), "{note}");
        assert!(note.contains("Ask your account admin"), "{note}");
    }

    #[test]
    fn empty_payload_still_available() {
        let limits = map_usage(&[]);
        assert_eq!(limits["available"], true);
        assert_eq!(limits["windows"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn b64_roundtrip() {
        // payload blobs decode; trailing '=' padding terminates.
        let raw = b"hello proto";
        let enc = "aGVsbG8gcHJvdG8=";
        assert_eq!(b64_decode(enc).unwrap(), raw);
    }
}
