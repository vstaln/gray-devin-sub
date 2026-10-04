//! Dynamic native catalog: `devin models list` is the live route table.
//!
//! Rows are two-space-indented `<id>  <display>  [<extra>]`; unindented
//! lines are family headers and `  aliases:` lines carry extra routable
//! ids. Context windows come only from the bracketed `<N>[KM] context`
//! segment — absent means `None`, never a guess. The listing is cached per
//! process; a CLI failure is an error, not a silent empty list.

use std::sync::{Mutex, OnceLock};

/// One routable model: id, display name, declared context window.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub display: String,
    pub context: Option<u32>,
}

/// `devin models list`, cached per process.
pub fn entries() -> Result<Vec<ModelEntry>, String> {
    static CACHE: OnceLock<Mutex<Option<Result<Vec<ModelEntry>, String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if let Some(hit) = cache.lock().ok().and_then(|c| c.clone()) {
        return hit;
    }
    let got = run_list().map(|text| parse_list(&text));
    if let Ok(mut c) = cache.lock() {
        *c = Some(got.clone());
    }
    got
}

fn run_list() -> Result<String, String> {
    let binary = crate::setup::resolve_command().ok_or_else(|| crate::setup::INSTALL_HINT.to_string())?;
    let mut child = std::process::Command::new(binary)
        .args(["models", "list"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .envs(crate::setup::child_env())
        .spawn()
        .map_err(|e| format!("devin models list: {e}"))?;
    // The listing is ~175KB — bigger than the pipe buffer. Drain stdout on
    // a reader thread while waiting: wait-then-read deadlocks the child on
    // a full pipe and always hits the deadline.
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "devin models list stdout unavailable".to_string())?;
    let reader = std::thread::spawn(move || {
        use std::io::Read;
        let mut s = String::new();
        let _ = stdout.read_to_string(&mut s);
        s
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err("devin models list timed out".into());
            }
            Err(e) => return Err(format!("devin models list: {e}")),
        }
    }
    Ok(reader.join().unwrap_or_default())
}

/// Parse `devin models list` text into routable entries.
pub fn parse_list(text: &str) -> Vec<ModelEntry> {
    let mut out: Vec<ModelEntry> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in text.lines() {
        if !line.starts_with("  ") {
            continue;
        }
        let body = line.trim_end();
        if body.trim_start().starts_with("aliases:") {
            // Extra routable ids Devin resolves itself; window unknown.
            for a in body.trim_start()["aliases:".len()..].split(|c: char| c == ',' || c == ' ') {
                let a = a.trim();
                if !a.is_empty() && seen.insert(a.to_string()) {
                    out.push(ModelEntry {
                        id: a.to_string(),
                        display: format!("{a} (alias)"),
                        context: None,
                    });
                }
            }
            continue;
        }
        let row = body.trim_start();
        // id followed by 2+ spaces, then display (+ optional [extra]).
        let split = row
            .char_indices()
            .zip(row.char_indices().skip(1))
            .find(|((_, a), (_, b))| a.is_whitespace() && b.is_whitespace())
            .map(|((i, _), _)| i);
        let Some(split) = split else { continue };
        let id = row[..split].trim();
        if id.is_empty() || id.contains(' ') {
            continue;
        }
        let rest = row[split..].trim();
        let (display, extra) = match rest.find('[') {
            Some(i) => (rest[..i].trim().to_string(), Some(&rest[i..])),
            None => (rest.to_string(), None),
        };
        if seen.insert(id.to_string()) {
            out.push(ModelEntry {
                id: id.to_string(),
                display: if display.is_empty() { id.to_string() } else { display },
                context: extra.and_then(context_of),
            });
        }
    }
    out
}

/// First bracket segment shaped `<N>[K|M] context` → window; else `None`.
fn context_of(bracketed: &str) -> Option<u32> {
    let inner = bracketed.trim_start_matches('[').trim_end_matches(']');
    for seg in inner.split(',') {
        let seg = seg.trim();
        let Some(num) = seg.strip_suffix(" context") else {
            continue;
        };
        let (digits, mult) = match num.chars().last() {
            Some('K') => (&num[..num.len() - 1], 1_000.0),
            Some('M') => (&num[..num.len() - 1], 1_000_000.0),
            _ => (num, 1.0),
        };
        if let Ok(v) = digits.parse::<f64>()
            && v > 0.0
        {
            return Some((v * mult) as u32);
        }
    }
    None
}

/// Native `--model` selection: ids pass through verbatim — the variant
/// (thought level, speed) lives in the id itself.
pub fn native_model(model: &str) -> String {
    model.to_string()
}

pub fn context_window(model: &str) -> Option<u32> {
    entries()
        .ok()?
        .into_iter()
        .find(|e| e.id == model)
        .and_then(|e| e.context)
}

/// Every routable id the CLI currently advertises.
pub fn all_ids() -> Vec<String> {
    entries()
        .map(|es| es.into_iter().map(|e| e.id).collect())
        .unwrap_or_default()
}

/// Display name for a route id.
pub fn display_name(id: &str) -> String {
    entries()
        .ok()
        .and_then(|es| es.into_iter().find(|e| e.id == id).map(|e| e.display))
        .unwrap_or_else(|| format!("{id} (Devin)"))
}

#[path = "catalog_tests.rs"]
#[cfg(test)]
mod tests;
