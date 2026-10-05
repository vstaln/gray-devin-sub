use super::*;
use crate::catalog::{Family, ModelEntry};

fn entry(id: &str, display: &str, context: Option<u32>) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        display: display.to_string(),
        context,
        family: Family::default(),
        alias: false,
    }
}

fn member(
    family_slug: &str,
    family_name: &str,
    id: &str,
    display: &str,
    context: Option<u32>,
) -> ModelEntry {
    let mut e = entry(id, display, context);
    e.family = Family {
        slug: family_slug.to_string(),
        name: family_name.to_string(),
    };
    e
}

#[test]
fn catalog_requires_cli() {
    // Without a resolvable `devin` binary the catalog is an error, never a
    // silently empty list. (When devin IS installed this asserts non-error
    // or error — either is a decided verdict.)
    let _ = catalog();
}

#[test]
fn collapse_folds_tier_families() {
    let entries = vec![
        member("swe-2", "SWE-2", "swe-2-high", "SWE-2 High", Some(262_000)),
        member(
            "swe-2",
            "SWE-2",
            "swe-2-medium",
            "SWE-2 Medium",
            Some(262_000),
        ),
        member("swe-2", "SWE-2", "swe-2-max", "SWE-2 Max", Some(262_000)),
        member(
            "swe-1.7-lightning",
            "SWE-1.7 Lightning",
            "swe-1-7-lightning",
            "SWE-1.7 Lightning Max",
            Some(202_752),
        ),
        member(
            "swe-1.7-lightning",
            "SWE-1.7 Lightning",
            "swe-1-7-lightning-medium",
            "SWE-1.7 Lightning Medium",
            Some(202_752),
        ),
        member(
            "claude-opus-5.5",
            "Claude Opus 5.5",
            "claude-opus-5-5-low",
            "Claude Opus 5.5 Low",
            Some(1_000_000),
        ),
        member(
            "claude-opus-5.5",
            "Claude Opus 5.5",
            "claude-opus-5-5-max",
            "Claude Opus 5.5 Max",
            Some(1_000_000),
        ),
        member(
            "claude-opus-5.5",
            "Claude Opus 5.5",
            "claude-opus-5-5-low-fast",
            "Claude Opus 5.5 Low Fast",
            Some(1_000_000),
        ),
        // A fusion pair: the trailing word is the sidekick's tier, not a
        // knob on the row — multi-word variants never collapse.
        member(
            "fusion",
            "Fusion",
            "fusion-claude-fable-5-1-low-sidekick-gpt-5-6-sol-high",
            "Fusion (Fable 5.1 Low + GPT-5.6 Sol High)",
            Some(1_000_000),
        ),
        member(
            "fusion",
            "Fusion",
            "fusion-claude-fable-5-1-low-sidekick-gpt-5-6-sol-max",
            "Fusion (Fable 5.1 Low + GPT-5.6 Sol Max)",
            Some(1_000_000),
        ),
        {
            let mut a = entry("swe", "swe (alias)", None);
            a.alias = true;
            a.family.slug = "swe-2".to_string();
            a
        },
        member("adaptive", "Adaptive", "adaptive", "Adaptive", None),
        entry("orphan-no-family", "Orphan", Some(100_000)),
    ];
    let models = collapse(entries);
    let get = |id: &str| models.iter().find(|m| m.id == id);

    // The pure-tier family is one row carrying its tiers as the knob,
    // named by its listing header.
    let swe2 = get("swe-2").expect("family row");
    assert_eq!(swe2.name, "SWE-2");
    assert_eq!(swe2.context_window, Some(262_000));
    assert_eq!(
        swe2.reasoning_efforts,
        vec!["medium", "high", "max"],
        "tiers keep canonical order, not listing order"
    );
    assert!(get("swe-2-max").is_none(), "tier members never list");

    // A family with a bare member does not collapse: the bare id is its
    // own tier, so nothing offers a knob.
    assert!(
        get("swe-1-7-lightning")
            .unwrap()
            .reasoning_efforts
            .is_empty()
    );
    assert!(
        get("swe-1-7-lightning-medium")
            .unwrap()
            .reasoning_efforts
            .is_empty()
    );

    // Partial collapse: pure members fold, the -fast variant stays.
    let opus = get("claude-opus-5-5").expect("collapsed family");
    assert_eq!(opus.name, "Claude Opus 5.5");
    assert_eq!(opus.reasoning_efforts, vec!["low", "max"]);
    assert!(
        get("claude-opus-5-5-low-fast")
            .unwrap()
            .reasoning_efforts
            .is_empty()
    );

    // Fusion ids never fold — their variant isn't a tier word.
    assert!(
        get("fusion-claude-fable-5-1-low-sidekick-gpt-5-6-sol-high")
            .unwrap()
            .reasoning_efforts
            .is_empty()
    );

    // Aliases, tier-less ids and unheadered rows stay, knobless.
    assert!(get("swe").unwrap().reasoning_efforts.is_empty());
    assert!(get("adaptive").unwrap().reasoning_efforts.is_empty());
    assert!(
        get("orphan-no-family")
            .unwrap()
            .reasoning_efforts
            .is_empty()
    );
}
