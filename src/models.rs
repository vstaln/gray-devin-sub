//! Live model catalog: `devin models list` IS the catalog — ids, display
//! names and context windows come from the CLI, never a pinned table.

use gray_plugin::{ProviderModel, ProviderModelCatalog};

use crate::catalog;

/// Devin ids already encode effort/speed in the id itself
/// (`gpt-6-sol-low`, `claude-opus-5-5-low-fast`): no separate effort knob.
const EFFORTS: &[&str] = &[];

/// The live catalog. A CLI failure is an error, not a silent empty list.
pub fn catalog() -> Result<ProviderModelCatalog, String> {
    let models = catalog::entries()?
        .into_iter()
        .map(|e| ProviderModel {
            name: e.display,
            context_window: e.context,
            reasoning_efforts: EFFORTS.iter().map(|s| s.to_string()).collect(),
            id: e.id,
        })
        .collect();
    Ok(ProviderModelCatalog { models })
}

#[path = "models_tests.rs"]
#[cfg(test)]
mod tests;
