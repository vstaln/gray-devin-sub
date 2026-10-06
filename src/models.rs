//! Live model catalog: `devin models list` IS the catalog — ids, display
//! names and context windows come from the CLI, never a pinned table.
//!
//! Devin bakes every knob into the id: thinking tier (`swe-2-max`), fast
//! lane (`claude-opus-5-5-high-fast`, `gpt-6-sol-high-priority`), no
//! thinking (`gpt-6-sol-none`), and Fusion pairings
//! (`fusion-<lead>-<tier>[-fast]-sidekick-<sidekick>[-priority]`). The
//! gray-facing list folds them back into one row per model:
//!
//! - **Plain rows**: ids sharing a parsed base ([`catalog::plain_groups`])
//!   become ONE row whose `variants` list every concrete id with its
//!   pinned effort (`off` for `-none`) and fast flag; `reasoning_efforts`
//!   is the union of those efforts. The host sends the picked variant's id
//!   verbatim; hosts that ignore `variants` send the row id + effort, which
//!   [`catalog::native_model`] maps back to the `<base>-<effort>` id.
//! - **Fusion**: every parseable pairing folds into ONE `fusion` row with
//!   `lead` / `sidekick` slots; each variant names its slot options in
//!   `parts`, its lead tier as `effort`, and the lead's fast lane as `fast`
//!   (the sidekick's `-priority` lane rides along).
//! - Everything else — `MODEL_*` legacy ids, `-1m` context products,
//!   `-thinking` products, single tier-less ids, aliases, unparseable
//!   fusion ids — stays its own row with no variants.

use std::collections::{BTreeMap, HashMap, HashSet};

use gray_plugin::{ModelSlot, ModelVariant, ProviderModel, ProviderModelCatalog, SlotOption};

use crate::catalog::{self, ModelEntry};

/// Row id (and slot-free model name) of the folded Fusion row.
pub const FUSION_ID: &str = "fusion";

/// The live catalog. A CLI failure is an error, not a silent empty list.
pub fn catalog() -> Result<ProviderModelCatalog, String> {
    Ok(ProviderModelCatalog {
        models: collapse(catalog::entries()?),
    })
}

/// One parsed Fusion pairing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionId {
    /// Lead base model id (`claude-opus-5-5`).
    pub lead: String,
    /// Lead tier, as an effort name (`off` for `none`).
    pub effort: Option<String>,
    /// Lead's `-fast` lane (the sidekick's `-priority` goes with it).
    pub fast: bool,
    /// Sidekick id with its tier, without `-priority` (`gpt-5-6-luna-high`).
    pub sidekick: String,
    /// Lead display minus variant words (`Claude Opus 5.5`), if listed.
    pub lead_name: Option<String>,
    /// Sidekick display minus `Thinking`/`Fast` (`GPT-5.6 Luna High`).
    pub sidekick_name: Option<String>,
}

/// Parse `fusion-<lead>[-<tier>][-fast]-sidekick-<sidekick>[-priority]`
/// plus its `Fusion (<lead display> + <sidekick display>)` listing name.
/// The id is authoritative; the display only supplies names. A
/// `-priority` sidekick under a non-fast lead is not a shape the listing
/// uses, so it is unparseable (kept as its own row) rather than guessed.
pub fn parse_fusion(id: &str, display: &str) -> Option<FusionId> {
    let body = id.strip_prefix("fusion-")?;
    let (lead_part, sk_part) = body.split_once("-sidekick-")?;
    if lead_part.is_empty() || sk_part.is_empty() || sk_part.contains("-sidekick-") {
        return None;
    }
    let mut lead = lead_part;
    let mut fast = false;
    if let Some(head) = lead.strip_suffix("-fast") {
        lead = head;
        fast = true;
    }
    let mut effort = None;
    if let Some((head, tail)) = lead.rsplit_once('-')
        && !head.is_empty()
        && (catalog::EFFORT_VARIANTS.contains(&tail) || tail == "none")
    {
        lead = head;
        effort = Some(if tail == "none" { "off" } else { tail }.to_string());
    }
    let (sidekick, priority) = match sk_part.strip_suffix("-priority") {
        Some(s) => (s, true),
        None => (sk_part, false),
    };
    if lead.is_empty() || sidekick.is_empty() || (priority && !fast) {
        return None;
    }
    let names = display
        .trim()
        .strip_prefix("Fusion (")
        .and_then(|d| d.strip_suffix(')'))
        .and_then(|d| d.split_once(" + "));
    let (lead_name, sidekick_name) = match names {
        Some((l, s)) => (Some(catalog::base_display(l)), Some(sidekick_display(s))),
        None => (None, None),
    };
    Some(FusionId {
        lead: lead.to_string(),
        effort,
        fast,
        sidekick: sidekick.to_string(),
        lead_name: lead_name.filter(|n| !n.is_empty()),
        sidekick_name: sidekick_name.filter(|n| !n.is_empty()),
    })
}

/// Sidekick display with its tier kept: drop a trailing `Fast`, then a
/// trailing `Thinking` (`GPT-5.6 Luna High Thinking Fast` →
/// `GPT-5.6 Luna High`).
fn sidekick_display(display: &str) -> String {
    let mut words: Vec<&str> = display.split_whitespace().collect();
    for w in ["fast", "thinking"] {
        if words.len() > 1 && words.last().is_some_and(|l| l.eq_ignore_ascii_case(w)) {
            words.pop();
        }
    }
    words.join(" ")
}

/// Listing rows → picker rows (see the module docs for the rules).
pub fn collapse(entries: Vec<ModelEntry>) -> Vec<ProviderModel> {
    let groups = catalog::plain_groups(&entries);
    let group_of: HashMap<&str, usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(i, g)| g.members.iter().map(move |(m, _, _)| (m.id.as_str(), i)))
        .collect();
    let group_names: HashMap<&str, String> =
        groups.iter().map(|g| (g.base.as_str(), g.name())).collect();

    // Fusion: parse every pairing up front; a duplicate (lead, sidekick,
    // effort, fast) key would make the pick ambiguous, so it stays a row.
    let mut fusion: Vec<(&ModelEntry, FusionId)> = Vec::new();
    let mut fusion_keys = HashSet::new();
    for e in entries
        .iter()
        .filter(|e| !e.alias && e.id.starts_with("fusion-"))
    {
        if let Some(f) = parse_fusion(&e.id, &e.display)
            && fusion_keys.insert((f.lead.clone(), f.sidekick.clone(), f.effort.clone(), f.fast))
        {
            fusion.push((e, f));
        }
    }
    let fusion_ids: HashSet<&str> = fusion.iter().map(|(e, _)| e.id.as_str()).collect();

    let mut models: Vec<ProviderModel> = Vec::new();
    let mut emitted: HashSet<String> = HashSet::new();
    for e in &entries {
        if fusion_ids.contains(e.id.as_str()) {
            if emitted.insert(FUSION_ID.to_string()) {
                models.push(fusion_row(&fusion, &group_names));
            }
            continue;
        }
        if let Some(&i) = group_of.get(e.id.as_str()) {
            let g = &groups[i];
            if emitted.insert(g.base.clone()) {
                models.push(ProviderModel {
                    id: g.base.clone(),
                    name: g.name(),
                    context_window: g.members.iter().filter_map(|(m, _, _)| m.context).max(),
                    reasoning_efforts: g.efforts(),
                    variants: g
                        .members
                        .iter()
                        .map(|(m, effort, fast)| ModelVariant {
                            id: m.id.clone(),
                            effort: effort.clone(),
                            fast: *fast,
                            parts: BTreeMap::new(),
                        })
                        .collect(),
                    slots: Vec::new(),
                });
            }
            continue;
        }
        if emitted.insert(e.id.clone()) {
            models.push(ProviderModel {
                id: e.id.clone(),
                name: e.display.clone(),
                context_window: e.context,
                reasoning_efforts: Vec::new(),
                variants: Vec::new(),
                slots: Vec::new(),
            });
        }
    }
    models
}

/// The single Fusion row: lead options are the distinct lead bases (named
/// by their plain row when one exists), sidekick options the distinct
/// tiered sidekicks, both in listing order.
fn fusion_row(
    fusion: &[(&ModelEntry, FusionId)],
    group_names: &HashMap<&str, String>,
) -> ProviderModel {
    let mut leads: Vec<SlotOption> = Vec::new();
    let mut sidekicks: Vec<SlotOption> = Vec::new();
    for (_, f) in fusion {
        if !leads.iter().any(|o| o.id == f.lead) {
            let name = group_names
                .get(f.lead.as_str())
                .cloned()
                .or_else(|| f.lead_name.clone())
                .unwrap_or_else(|| f.lead.clone());
            leads.push(SlotOption {
                id: f.lead.clone(),
                name,
            });
        }
        if !sidekicks.iter().any(|o| o.id == f.sidekick) {
            sidekicks.push(SlotOption {
                id: f.sidekick.clone(),
                name: f
                    .sidekick_name
                    .clone()
                    .unwrap_or_else(|| f.sidekick.clone()),
            });
        }
    }
    let reasoning_efforts = catalog::EFFORT_ORDER
        .iter()
        .filter(|t| fusion.iter().any(|(_, f)| f.effort.as_deref() == Some(**t)))
        .map(|t| t.to_string())
        .collect();
    ProviderModel {
        id: FUSION_ID.to_string(),
        name: "Fusion".to_string(),
        context_window: fusion.iter().filter_map(|(e, _)| e.context).max(),
        reasoning_efforts,
        variants: fusion
            .iter()
            .map(|(e, f)| ModelVariant {
                id: e.id.clone(),
                effort: f.effort.clone(),
                fast: f.fast,
                parts: BTreeMap::from([
                    ("lead".to_string(), f.lead.clone()),
                    ("sidekick".to_string(), f.sidekick.clone()),
                ]),
            })
            .collect(),
        slots: vec![
            ModelSlot {
                key: "lead".to_string(),
                label: "Lead".to_string(),
                options: leads,
            },
            ModelSlot {
                key: "sidekick".to_string(),
                label: "Sidekick".to_string(),
                options: sidekicks,
            },
        ],
    }
}

#[path = "models_tests.rs"]
#[cfg(test)]
mod tests;
