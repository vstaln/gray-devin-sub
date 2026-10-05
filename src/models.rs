//! Live model catalog: `devin models list` IS the catalog — ids, display
//! names and context windows come from the CLI, never a pinned table.
//!
//! Effort tiers live in the id (`swe-2-max`, `claude-opus-5-5-low`), so the
//! gray-facing list collapses each pure-tier family into ONE entry — the
//! picker shows `swe-2`, not `SWE-2 Medium/High/Max` — and declares the
//! family's tiers as `reasoning_efforts` so `/thinking` offers exactly the
//! levels that exist. `chat` maps `<family>` + request effort back to the
//! `<family>-<effort>` id ([`catalog::native_model`]). Everything else —
//! `-fast` speed variants, fusion ids, bare family ids, aliases — stays a
//! separate row with an empty effort list: its tier is fixed, so no knob.

use std::collections::{HashMap, HashSet};

use gray_plugin::{ProviderModel, ProviderModelCatalog};

use crate::catalog::{self, ModelEntry};

/// The live catalog. A CLI failure is an error, not a silent empty list.
pub fn catalog() -> Result<ProviderModelCatalog, String> {
    Ok(ProviderModelCatalog {
        models: collapse(catalog::entries()?),
    })
}

/// Rows → picker entries: a family whose members differ only by a tier
/// variant folds into one `<base>` row carrying the tier list as
/// `reasoning_efforts`; every other row keeps its id and advertises no
/// effort knob.
fn collapse(entries: Vec<ModelEntry>) -> Vec<ProviderModel> {
    let ids: HashSet<&str> = entries.iter().map(|e| e.id.as_str()).collect();
    let fams = catalog::families(&entries);

    // member id → its collapsible family (only tier-variant members fold).
    let member_fam: HashMap<&str, &catalog::FamilyMembers> = fams
        .iter()
        .filter(|f| catalog::family_efforts(f, &ids).is_some())
        .flat_map(|f| {
            f.members
                .iter()
                .filter(|(_, v)| catalog::EFFORT_VARIANTS.contains(v))
                .map(move |(m, _)| (m.id.as_str(), f))
        })
        .collect();

    let mut models: Vec<ProviderModel> = Vec::new();
    let mut emitted: HashSet<&str> = HashSet::new();
    for e in &entries {
        match member_fam.get(e.id.as_str()) {
            Some(fam) => {
                if emitted.insert(fam.base.as_str()) {
                    models.push(ProviderModel {
                        name: if fam.name.is_empty() {
                            fam.base.clone()
                        } else {
                            fam.name.to_string()
                        },
                        context_window: fam.members.iter().filter_map(|(m, _)| m.context).max(),
                        reasoning_efforts: catalog::family_efforts(fam, &ids).unwrap_or_default(),
                        id: fam.base.clone(),
                    });
                }
                // later tier members fold into the emitted family row
            }
            None => {
                if emitted.insert(e.id.as_str()) {
                    models.push(ProviderModel {
                        name: e.display.clone(),
                        context_window: e.context,
                        reasoning_efforts: Vec::new(),
                        id: e.id.clone(),
                    });
                }
            }
        }
    }
    models
}

#[path = "models_tests.rs"]
#[cfg(test)]
mod tests;
