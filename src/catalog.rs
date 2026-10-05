//! Dynamic native catalog: `devin models list` is the live route table.
//!
//! Rows are two-space-indented `<id>  <display>  [<extra>]`; unindented
//! lines are family headers and `  aliases:` lines carry extra routable
//! ids. Context windows come only from the bracketed `<N>[KM] context`
//! segment — absent means `None`, never a guess. The listing is cached per
//! process; a CLI failure is an error, not a silent empty list.

use std::sync::{Mutex, OnceLock};

/// The family a row was listed under — parsed from its section header
/// `Display Name (slug)`. The slug is a routable fuzzy name the
/// `--model` matcher accepts.
#[derive(Debug, Clone, Default)]
pub struct Family {
    pub slug: String,
    pub name: String,
}

/// One routable model: id, display name, declared context window, its
/// listing family, and whether it came from an `aliases:` line (extra
/// routable ids, not real members).
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub display: String,
    pub context: Option<u32>,
    pub family: Family,
    pub alias: bool,
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
    let binary =
        crate::setup::resolve_command().ok_or_else(|| crate::setup::INSTALL_HINT.to_string())?;
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

/// Parse `devin models list` text into routable entries. Unindented lines
/// are family headers (`SWE-2 (swe-2)`); the parenthesized slug is the
/// routable family name the `--model` fuzzy matcher accepts.
pub fn parse_list(text: &str) -> Vec<ModelEntry> {
    let mut out: Vec<ModelEntry> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut family = Family::default();
    for line in text.lines() {
        if !line.starts_with("  ") {
            let h = line.trim();
            // Blank separators don't end a family — rows after them still
            // belong to the last header.
            if h.is_empty() {
                continue;
            }
            // Header: `Display Name (slug)`. Anything unindented that
            // doesn't parse resets the family — its rows don't belong to
            // the previous section.
            family = Family::default();
            if let Some((name, slug)) = h.rsplit_once('(')
                && let Some(slug) = slug.strip_suffix(')')
                && slug
                    .trim()
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && !slug.trim().is_empty()
            {
                family = Family {
                    slug: slug.trim().to_string(),
                    name: name.trim().to_string(),
                };
            }
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
                        family: family.clone(),
                        alias: true,
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
                display: if display.is_empty() {
                    id.to_string()
                } else {
                    display
                },
                context: extra.and_then(context_of),
                family: family.clone(),
                alias: false,
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

/// Variant words that name a thinking tier — the picker's vocabulary
/// minus `off` (`off` never crosses the wire, and a `-none` row is a
/// separate non-reasoning product). Any other variant — `fast`,
/// `priority`, `turbo`, a sidekick id — is a product trait and keeps its
/// own row.
pub const EFFORT_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// Rows under one listing header, expressed as `<base>-<variant>`:
/// `base` is the members' shared id prefix, and each member's variant is
/// what follows it.
pub struct FamilyMembers<'a> {
    /// Header slug — the routable family name.
    pub slug: &'a str,
    /// Header display text.
    pub name: &'a str,
    /// Longest common member-id prefix clipped at a `-` boundary.
    pub base: String,
    /// (row, variant) in listing order; an empty variant means the row
    /// IS the base — a bare routable member.
    pub members: Vec<(&'a ModelEntry, &'a str)>,
}

/// Group non-alias rows by listing family and compute each family's
/// `<base>-<variant>` shape. Rows listed before any header are ignored.
pub fn families(entries: &[ModelEntry]) -> Vec<FamilyMembers<'_>> {
    let mut order: Vec<&str> = Vec::new();
    let mut grouped: Vec<Vec<&ModelEntry>> = Vec::new();
    for e in entries
        .iter()
        .filter(|e| !e.alias && !e.family.slug.is_empty())
    {
        match order.iter().position(|s| *s == e.family.slug) {
            Some(i) => grouped[i].push(e),
            None => {
                order.push(e.family.slug.as_str());
                grouped.push(vec![e]);
            }
        }
    }
    order
        .into_iter()
        .zip(grouped)
        .map(|(slug, members)| {
            let base = common_base(&members);
            FamilyMembers {
                slug,
                name: members
                    .first()
                    .map(|m| m.family.name.as_str())
                    .unwrap_or_default(),
                members: members
                    .iter()
                    .map(|m| (*m, variant_of(&m.id, &base)))
                    .collect(),
                base,
            }
        })
        .collect()
}

/// Longest common member-id prefix. A member that IS the prefix is kept
/// whole (it's the family's bare row); otherwise clip at the last `-`.
fn common_base(members: &[&ModelEntry]) -> String {
    let mut p = members.first().map(|m| m.id.clone()).unwrap_or_default();
    for m in &members[1..] {
        let n = p
            .bytes()
            .zip(m.id.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        p.truncate(n);
    }
    if members.iter().any(|m| m.id == p) {
        // A member ending in a tier word is itself a variant row, not the
        // base: solo `claude-opus-5-5-medium` still yields
        // `claude-opus-5-5`. Bare rows (`swe-1-7-lightning`) keep whole.
        if let Some((head, tail)) = p.rsplit_once('-')
            && EFFORT_VARIANTS.contains(&tail)
            && !head.is_empty()
        {
            return head.to_string();
        }
        return p;
    }
    match p.rfind('-') {
        Some(i) => p.truncate(i),
        None => p.clear(),
    }
    p
}

fn variant_of<'a>(id: &'a str, base: &str) -> &'a str {
    if id == base {
        return "";
    }
    id.strip_prefix(base)
        .and_then(|s| s.strip_prefix('-'))
        .unwrap_or(id)
}

/// A family folds into one picker row when it has tier variants, no bare
/// base row, and `base` isn't a routable id elsewhere in the catalog.
/// Returns the declared efforts in picker order, or `None` when the
/// family must stay as individual rows.
pub fn family_efforts(
    fam: &FamilyMembers<'_>,
    ids: &std::collections::HashSet<&str>,
) -> Option<Vec<String>> {
    if fam.base.is_empty()
        // `SWE-2 (swe-2)` owns `swe-2-*` tier rows; `Fusion (fusion)`
        // heads pairings — `...-sol-high`'s tail is the sidekick's tier
        // baked into a product id, not a knob on one row.
        || fam.slug.replace('.', "-") != fam.base
        || ids.contains(fam.base.as_str())
        || fam.members.iter().any(|(_, v)| v.is_empty())
    {
        return None;
    }
    let tiers: Vec<String> = EFFORT_VARIANTS
        .iter()
        .filter(|t| fam.members.iter().any(|(_, v)| *v == **t))
        .map(|t| t.to_string())
        .collect();
    if tiers.is_empty() { None } else { Some(tiers) }
}

/// Native `--model` selection. A real catalog id passes through verbatim:
/// the variant (tier, speed) lives in the id itself, so a stale
/// `swe-2-max` pick stays `swe-2-max` whatever the request's effort says.
/// A collapsed family id (`swe-2`) resolves to its `<base>-<effort>`
/// member when the request asks for a tier the family has; otherwise it
/// falls back to the header slug — the documented routable family name —
/// which lets Devin pick the family's default tier.
pub fn native_model(model: &str, effort: Option<&str>) -> String {
    let Ok(entries) = entries() else {
        return model.to_string();
    };
    if entries.iter().any(|e| e.id == model) {
        return model.to_string();
    }
    let ids: std::collections::HashSet<&str> =
        entries.iter().map(|e| e.id.as_str()).collect();
    for fam in families(&entries) {
        if fam.base == model && family_efforts(&fam, &ids).is_some() {
            if let Some(tier) = effort
                && let Some((m, _)) = fam.members.iter().find(|(_, v)| *v == tier)
            {
                return m.id.clone();
            }
            return fam.slug.to_string();
        }
    }
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
