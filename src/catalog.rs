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
    type CatalogResult = Result<Vec<ModelEntry>, String>;
    static CACHE: OnceLock<Mutex<Option<CatalogResult>>> = OnceLock::new();
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
            for a in body.trim_start()["aliases:".len()..].split([',', ' ']) {
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

/// Id-suffix words that name a thinking tier, in picker order. `none`
/// is listed separately: a `-none` id is the effort-`off` variant.
pub const EFFORT_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// Picker order for declared efforts: `off` (a `-none` id) first, then
/// the tiers.
pub const EFFORT_ORDER: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Id suffixes that select the fast lane: Anthropic-style `-fast` and
/// OpenAI-style `-priority` (listed as "... Fast").
const FAST_SUFFIXES: &[&str] = &["fast", "priority"];

/// Tier word → effort name: `none` is the picker's `off`.
fn effort_name(tier: &str) -> String {
    if tier == "none" {
        "off".to_string()
    } else {
        tier.to_string()
    }
}

/// The thinking tier a display string names, in id vocabulary
/// (`none`, `low`, …, `xhigh`, `max`): `GPT-6 Sol High Thinking Fast` →
/// `high`, `GPT-5.4 No Thinking` / `Inkling None` → `none`,
/// `Inkling X-High` → `xhigh`. `None` when the display names no tier
/// (`SWE-1.6`, `Claude Opus 4.6 Thinking`).
pub fn display_tier(display: &str) -> Option<&'static str> {
    let mut words: Vec<&str> = display.split_whitespace().collect();
    if words.last().is_some_and(|w| w.eq_ignore_ascii_case("fast")) {
        words.pop();
    }
    if words.len() >= 2
        && words[words.len() - 2].eq_ignore_ascii_case("no")
        && words[words.len() - 1].eq_ignore_ascii_case("thinking")
    {
        return Some("none");
    }
    if words
        .last()
        .is_some_and(|w| w.eq_ignore_ascii_case("thinking"))
    {
        words.pop();
    }
    let last = words.last()?.to_ascii_lowercase().replace('-', "");
    EFFORT_VARIANTS
        .iter()
        .chain(std::iter::once(&"none"))
        .find(|t| **t == last)
        .copied()
}

/// Display text minus its variant words (tier, `Thinking`, `Fast`):
/// `GPT-6 Sol High Thinking Fast` → `GPT-6 Sol`, `SWE-1.6 Fast` →
/// `SWE-1.6`.
pub fn base_display(display: &str) -> String {
    let mut words: Vec<&str> = display.split_whitespace().collect();
    if words.last().is_some_and(|w| w.eq_ignore_ascii_case("fast")) {
        words.pop();
    }
    if words
        .last()
        .is_some_and(|w| w.eq_ignore_ascii_case("thinking"))
    {
        words.pop();
        if words.last().is_some_and(|w| w.eq_ignore_ascii_case("no")) {
            words.pop();
        }
    }
    if words.len() > 1
        && let Some(last) = words.last()
    {
        let l = last.to_ascii_lowercase().replace('-', "");
        if EFFORT_VARIANTS.contains(&l.as_str()) || l == "none" {
            words.pop();
        }
    }
    words.join(" ")
}

fn has_word(display: &str, word: &str) -> bool {
    display
        .split_whitespace()
        .any(|w| w.eq_ignore_ascii_case(word))
}

/// One parsed plain id: `<base>[-<tier>][-fast|-priority]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlainId {
    pub base: String,
    /// Effort the id is pinned to (`off` for `-none`); `None` = no tier
    /// in the id or its display.
    pub effort: Option<String>,
    pub fast: bool,
}

/// Parse a plain (non-fusion) id. Every stripped suffix must be confirmed
/// by the display text — `-fast`/`-priority` by a `Fast` word, a tier by
/// the display's tier word — so a product name that merely ends in a tier
/// word is never split. A bare id whose display names a tier
/// (`swe-1-7-lightning` = "SWE-1.7 Lightning Max") is pinned to it.
/// `None` for ids outside the lowercase `a-z0-9-` shape (`MODEL_*`
/// legacy ids) and for `fusion-*` pairings.
pub fn parse_plain(id: &str, display: &str) -> Option<PlainId> {
    if id.is_empty()
        || id.starts_with("fusion-")
        || !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return None;
    }
    let mut rest = id;
    let mut fast = false;
    if let Some((head, tail)) = rest.rsplit_once('-')
        && FAST_SUFFIXES.contains(&tail)
        && !head.is_empty()
        && has_word(display, "fast")
    {
        rest = head;
        fast = true;
    }
    let shown = display_tier(display);
    let mut effort = None;
    if let Some((head, tail)) = rest.rsplit_once('-')
        && !head.is_empty()
        && (EFFORT_VARIANTS.contains(&tail) || tail == "none")
        && shown == Some(tail)
    {
        rest = head;
        effort = Some(effort_name(tail));
    } else if let Some(t) = shown {
        effort = Some(effort_name(t));
    }
    Some(PlainId {
        base: rest.to_string(),
        effort,
        fast,
    })
}

/// Plain ids sharing one parsed base, across listing families
/// (`swe-1-6` + `swe-1-6-fast` are one model with a fast lane).
pub struct PlainGroup<'a> {
    pub base: String,
    /// (row, effort, fast) in listing order.
    pub members: Vec<(&'a ModelEntry, Option<String>, bool)>,
}

impl PlainGroup<'_> {
    /// Picker name: the listing header whose slug is this base
    /// (`Claude Opus 5.5`), else the first member's display minus its
    /// variant words.
    pub fn name(&self) -> String {
        if let Some((m, _, _)) = self
            .members
            .iter()
            .find(|(m, _, _)| !m.family.name.is_empty() && self.slug_matches(&m.family))
        {
            return m.family.name.clone();
        }
        self.members
            .first()
            .map(|(m, _, _)| base_display(&m.display))
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| self.base.clone())
    }

    /// The routable family slug Devin resolves to its default tier: the
    /// header slug matching this base, else the first member's header.
    pub fn slug(&self) -> Option<&str> {
        self.members
            .iter()
            .find(|(m, _, _)| self.slug_matches(&m.family))
            .or_else(|| self.members.first())
            .map(|(m, _, _)| m.family.slug.as_str())
            .filter(|s| !s.is_empty())
    }

    fn slug_matches(&self, family: &Family) -> bool {
        !family.slug.is_empty() && family.slug.replace('.', "-") == self.base
    }

    /// The efforts members declare, in picker order.
    pub fn efforts(&self) -> Vec<String> {
        EFFORT_ORDER
            .iter()
            .filter(|t| {
                self.members
                    .iter()
                    .any(|(_, e, _)| e.as_deref() == Some(**t))
            })
            .map(|t| t.to_string())
            .collect()
    }
}

/// Group the routable plain ids by parsed base and keep the groups that
/// fold into one picker row: members carry a variant (an effort or the
/// fast lane), each (effort, fast) pair names exactly one id, and the
/// base isn't some other routable id. Aliases and unparseable ids
/// (`MODEL_*`, `fusion-*`) never group.
pub fn plain_groups(entries: &[ModelEntry]) -> Vec<PlainGroup<'_>> {
    let ids: std::collections::HashSet<&str> = entries.iter().map(|e| e.id.as_str()).collect();
    let mut groups: Vec<PlainGroup<'_>> = Vec::new();
    for e in entries.iter().filter(|e| !e.alias) {
        let Some(p) = parse_plain(&e.id, &e.display) else {
            continue;
        };
        match groups.iter_mut().find(|g| g.base == p.base) {
            Some(g) => g.members.push((e, p.effort, p.fast)),
            None => groups.push(PlainGroup {
                base: p.base,
                members: vec![(e, p.effort, p.fast)],
            }),
        }
    }
    groups.retain(|g| {
        let varied = g.members.len() > 1 || g.members.iter().any(|(_, e, f)| e.is_some() || *f);
        let mut keys = std::collections::HashSet::new();
        let unique = g
            .members
            .iter()
            .all(|(_, e, f)| keys.insert((e.clone(), *f)));
        let base_free =
            !ids.contains(g.base.as_str()) || g.members.iter().any(|(m, _, _)| m.id == g.base);
        varied && unique && base_free
    });
    groups
}

/// Native `--model` selection. A real catalog id passes through verbatim:
/// the variant (tier, speed) lives in the id itself, so a variant id the
/// picker resolved — or a stale `swe-2-max` pick — stays as sent whatever
/// the request's effort says. A folded row id that isn't itself routable
/// (`swe-2`, from older configs or hosts that don't read `variants`)
/// resolves to its non-fast member pinned to the request's effort
/// (`swe-2` + `max` → `swe-2-max`, `gpt-6-sol` + `off` → `gpt-6-sol-none`);
/// otherwise it falls back to the header slug — the documented routable
/// family name — which lets Devin pick the family's default tier. `fusion`
/// with no resolved pairing is the Fusion header slug and passes through.
pub fn native_model(model: &str, effort: Option<&str>) -> String {
    let Ok(entries) = entries() else {
        return model.to_string();
    };
    resolve_native(&entries, model, effort)
}

/// [`native_model`] over an explicit listing.
pub fn resolve_native(entries: &[ModelEntry], model: &str, effort: Option<&str>) -> String {
    if entries.iter().any(|e| e.id == model) {
        return model.to_string();
    }
    let Some(group) = plain_groups(entries).into_iter().find(|g| g.base == model) else {
        return model.to_string();
    };
    let want = effort.map(|t| if t == "none" { "off" } else { t });
    if let Some(t) = want
        && let Some((m, _, _)) = group
            .members
            .iter()
            .find(|(_, e, fast)| !fast && e.as_deref() == Some(t))
    {
        return m.id.clone();
    }
    group.slug().unwrap_or(model).to_string()
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
