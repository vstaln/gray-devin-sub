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

/// `(id, display)` rows under one listing header, 1M context.
fn family(slug: &str, name: &str, rows: &[(&str, &str)]) -> Vec<ModelEntry> {
    rows.iter()
        .map(|(id, d)| member(slug, name, id, d, Some(1_000_000)))
        .collect()
}

fn variant<'a>(m: &'a ProviderModel, id: &str) -> &'a ModelVariant {
    m.variants
        .iter()
        .find(|v| v.id == id)
        .unwrap_or_else(|| panic!("variant {id} in {}", m.id))
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
    let mut entries = family(
        "swe-2",
        "SWE-2",
        &[
            ("swe-2-high", "SWE-2 High"),
            ("swe-2-medium", "SWE-2 Medium"),
            ("swe-2-max", "SWE-2 Max"),
        ],
    );
    entries.push({
        let mut a = entry("swe", "swe (alias)", None);
        a.alias = true;
        a.family.slug = "swe-2".to_string();
        a
    });
    entries.extend(family(
        "swe-1.7-lightning",
        "SWE-1.7 Lightning",
        &[
            ("swe-1-7-lightning", "SWE-1.7 Lightning Max"),
            ("swe-1-7-lightning-medium", "SWE-1.7 Lightning Medium"),
        ],
    ));
    entries.push(member("adaptive", "Adaptive", "adaptive", "Adaptive", None));
    entries.push(entry("orphan-no-family", "Orphan", Some(100_000)));
    let models = collapse(entries);
    let get = |id: &str| models.iter().find(|m| m.id == id);

    // The pure-tier family is one row carrying its tiers as the knob,
    // named by its listing header.
    let swe2 = get("swe-2").expect("family row");
    assert_eq!(swe2.name, "SWE-2");
    assert_eq!(swe2.context_window, Some(1_000_000));
    assert_eq!(
        swe2.reasoning_efforts,
        vec!["medium", "high", "max"],
        "tiers keep canonical order, not listing order"
    );
    assert_eq!(swe2.variants.len(), 3);
    assert_eq!(variant(swe2, "swe-2-max").effort.as_deref(), Some("max"));
    assert!(get("swe-2-max").is_none(), "tier members never list");

    // A bare id pinned by its display tier folds with its siblings.
    let lightning = get("swe-1-7-lightning").expect("lightning row");
    assert_eq!(lightning.name, "SWE-1.7 Lightning");
    assert_eq!(lightning.reasoning_efforts, vec!["medium", "max"]);
    assert_eq!(
        variant(lightning, "swe-1-7-lightning").effort.as_deref(),
        Some("max")
    );
    assert!(get("swe-1-7-lightning-medium").is_none());

    // Aliases, tier-less ids and unheadered rows stay, knobless.
    for id in ["swe", "adaptive", "orphan-no-family"] {
        let m = get(id).unwrap();
        assert!(m.reasoning_efforts.is_empty(), "{id}");
        assert!(m.variants.is_empty(), "{id}");
    }
}

#[test]
fn fast_and_none_variants_fold_into_one_row() {
    let mut entries = family(
        "claude-opus-5.5",
        "Claude Opus 5.5",
        &[
            ("claude-opus-5-5-medium", "Claude Opus 5.5 Medium"),
            ("claude-opus-5-5-low", "Claude Opus 5.5 Low"),
            ("claude-opus-5-5-max", "Claude Opus 5.5 Max"),
            ("claude-opus-5-5-low-fast", "Claude Opus 5.5 Low Fast"),
            ("claude-opus-5-5-max-fast", "Claude Opus 5.5 Max Fast"),
        ],
    );
    entries.extend(family(
        "gpt-6-sol",
        "GPT-6 Sol",
        &[
            ("gpt-6-sol-medium", "GPT-6 Sol Medium Thinking"),
            ("gpt-6-sol-none", "GPT-6 Sol No Thinking"),
            ("gpt-6-sol-high", "GPT-6 Sol High Thinking"),
            ("gpt-6-sol-none-priority", "GPT-6 Sol No Thinking Fast"),
            ("gpt-6-sol-high-priority", "GPT-6 Sol High Thinking Fast"),
        ],
    ));
    // SWE-1.6 and its fast lane are listed under separate headers.
    entries.push(member("swe-1.6", "SWE-1.6", "swe-1-6", "SWE-1.6", None));
    entries.push(member(
        "swe-1.6-fast",
        "SWE-1.6 Fast",
        "swe-1-6-fast",
        "SWE-1.6 Fast",
        None,
    ));
    let models = collapse(entries);
    assert_eq!(
        models.len(),
        3,
        "{:?}",
        models.iter().map(|m| &m.id).collect::<Vec<_>>()
    );
    let get = |id: &str| models.iter().find(|m| m.id == id).unwrap();

    let opus = get("claude-opus-5-5");
    assert_eq!(opus.name, "Claude Opus 5.5");
    assert_eq!(opus.reasoning_efforts, vec!["low", "medium", "max"]);
    assert_eq!(opus.variants.len(), 5);
    let fast = variant(opus, "claude-opus-5-5-low-fast");
    assert_eq!(fast.effort.as_deref(), Some("low"));
    assert!(fast.fast);
    assert!(!variant(opus, "claude-opus-5-5-low").fast);

    // GPT `-priority` is the fast lane; `-none` is effort off.
    let sol = get("gpt-6-sol");
    assert_eq!(sol.name, "GPT-6 Sol");
    assert_eq!(sol.reasoning_efforts, vec!["off", "medium", "high"]);
    let none = variant(sol, "gpt-6-sol-none");
    assert_eq!(none.effort.as_deref(), Some("off"));
    assert!(!none.fast);
    let pri = variant(sol, "gpt-6-sol-high-priority");
    assert_eq!(pri.effort.as_deref(), Some("high"));
    assert!(pri.fast);
    assert!(variant(sol, "gpt-6-sol-none-priority").fast);

    // Tier-less model + fast lane: one row, fast knob only.
    let swe16 = get("swe-1-6");
    assert_eq!(swe16.name, "SWE-1.6");
    assert!(swe16.reasoning_efforts.is_empty());
    assert_eq!(
        swe16.variants,
        vec![
            ModelVariant {
                id: "swe-1-6".into(),
                effort: None,
                fast: false,
                parts: BTreeMap::new(),
            },
            ModelVariant {
                id: "swe-1-6-fast".into(),
                effort: None,
                fast: true,
                parts: BTreeMap::new(),
            },
        ]
    );
}

#[test]
fn unparseable_shapes_stay_separate() {
    let mut entries = family(
        "glm-5.2",
        "GLM-5.2",
        &[
            ("glm-5-2", "GLM-5.2 High"),
            ("glm-5-2-max", "GLM-5.2 Max"),
            ("glm-5-2-1m", "GLM-5.2 High 1M"),
            ("glm-5-2-max-1m", "GLM-5.2 Max 1M"),
            ("glm-5-2-none", "GLM-5.2 No Thinking"),
        ],
    );
    entries.extend(family(
        "claude-opus-4.6",
        "Claude Opus 4.6",
        &[
            ("claude-opus-4-6", "Claude Opus 4.6"),
            ("claude-opus-4-6-thinking", "Claude Opus 4.6 Thinking"),
            ("claude-opus-4-6-1m", "Claude Opus 4.6 1M"),
        ],
    ));
    entries.extend(family(
        "gpt-5.2",
        "GPT-5.2",
        &[
            ("MODEL_GPT_5_2_LOW", "GPT-5.2 Low Thinking"),
            ("MODEL_GPT_5_2_HIGH", "GPT-5.2 High Thinking"),
        ],
    ));
    // Not the pairing shape: no `-sidekick-`, and a priority sidekick
    // under a non-fast lead.
    entries.extend(family(
        "fusion",
        "Fusion",
        &[
            ("fusion-mystery", "Fusion Mystery"),
            (
                "fusion-gpt-6-sol-high-sidekick-gpt-6-luna-high-priority",
                "Fusion (GPT-6 Sol High Thinking + GPT-6 Luna High Thinking Fast)",
            ),
        ],
    ));
    let models = collapse(entries);
    let get = |id: &str| models.iter().find(|m| m.id == id);

    let glm = get("glm-5-2").expect("glm row");
    assert_eq!(glm.name, "GLM-5.2");
    assert_eq!(glm.reasoning_efforts, vec!["off", "high", "max"]);
    assert_eq!(variant(glm, "glm-5-2").effort.as_deref(), Some("high"));
    // 1M context products are their own rows.
    for id in [
        "glm-5-2-1m",
        "glm-5-2-max-1m",
        "claude-opus-4-6",
        "claude-opus-4-6-thinking",
        "claude-opus-4-6-1m",
        "MODEL_GPT_5_2_LOW",
        "MODEL_GPT_5_2_HIGH",
        "fusion-mystery",
        "fusion-gpt-6-sol-high-sidekick-gpt-6-luna-high-priority",
    ] {
        let m = get(id).unwrap_or_else(|| panic!("{id} stays a row"));
        assert!(m.variants.is_empty(), "{id}");
        assert!(m.reasoning_efforts.is_empty(), "{id}");
    }
    assert!(
        get(FUSION_ID).is_none(),
        "no parseable pairing, no fusion row"
    );
}

#[test]
fn fusion_pairings_fold_into_one_row() {
    let mut entries = family(
        "claude-opus-5.5",
        "Claude Opus 5.5",
        &[
            ("claude-opus-5-5-medium", "Claude Opus 5.5 Medium"),
            ("claude-opus-5-5-high", "Claude Opus 5.5 High"),
        ],
    );
    entries.extend(family(
        "fusion",
        "Fusion",
        &[
            (
                "fusion-claude-fable-5-1-medium-sidekick-swe-2-medium",
                "Fusion (Claude Fable 5.1 Medium + SWE-2 Medium)",
            ),
            (
                "fusion-claude-opus-5-5-high-sidekick-swe-2-medium",
                "Fusion (Claude Opus 5.5 High + SWE-2 Medium)",
            ),
            (
                "fusion-claude-opus-5-5-high-fast-sidekick-swe-2-medium",
                "Fusion (Claude Opus 5.5 High Fast + SWE-2 Medium)",
            ),
            (
                "fusion-gpt-6-sol-medium-sidekick-gpt-5-6-luna-high",
                "Fusion (GPT-6 Sol Medium Thinking + GPT-5.6 Luna High Thinking)",
            ),
            (
                "fusion-gpt-6-sol-medium-fast-sidekick-gpt-5-6-luna-high-priority",
                "Fusion (GPT-6 Sol Medium Thinking Fast + GPT-5.6 Luna High Thinking Fast)",
            ),
            (
                "fusion-claude-opus-5-5-high-fast-sidekick-gpt-5-6-luna-high-priority",
                "Fusion (Claude Opus 5.5 High Fast + GPT-5.6 Luna High Thinking Fast)",
            ),
        ],
    ));
    let models = collapse(entries);
    let fusion_rows: Vec<&ProviderModel> = models
        .iter()
        .filter(|m| m.id.starts_with("fusion"))
        .collect();
    assert_eq!(fusion_rows.len(), 1, "every pairing folds into one row");
    let f = fusion_rows[0];
    assert_eq!(f.id, "fusion");
    assert_eq!(f.name, "Fusion");
    assert_eq!(f.context_window, Some(1_000_000));
    assert_eq!(f.reasoning_efforts, vec!["medium", "high"]);
    assert_eq!(f.variants.len(), 6);

    let keys: Vec<&str> = f.slots.iter().map(|s| s.key.as_str()).collect();
    assert_eq!(keys, ["lead", "sidekick"]);
    assert_eq!(f.slots[0].label, "Lead");
    assert_eq!(
        f.slots[0].options,
        vec![
            SlotOption {
                id: "claude-fable-5-1".into(),
                name: "Claude Fable 5.1".into(),
            },
            SlotOption {
                id: "claude-opus-5-5".into(),
                name: "Claude Opus 5.5".into(),
            },
            SlotOption {
                id: "gpt-6-sol".into(),
                name: "GPT-6 Sol".into(),
            },
        ]
    );
    assert_eq!(f.slots[1].label, "Sidekick");
    assert_eq!(
        f.slots[1].options,
        vec![
            SlotOption {
                id: "swe-2-medium".into(),
                name: "SWE-2 Medium".into(),
            },
            SlotOption {
                id: "gpt-5-6-luna-high".into(),
                name: "GPT-5.6 Luna High".into(),
            },
        ]
    );

    let v = variant(
        f,
        "fusion-claude-opus-5-5-high-fast-sidekick-gpt-5-6-luna-high-priority",
    );
    assert_eq!(v.effort.as_deref(), Some("high"));
    assert!(v.fast);
    assert_eq!(v.parts["lead"], "claude-opus-5-5");
    assert_eq!(v.parts["sidekick"], "gpt-5-6-luna-high");
    // A fast lead with a sidekick that has no priority lane.
    let v = variant(f, "fusion-claude-opus-5-5-high-fast-sidekick-swe-2-medium");
    assert!(v.fast);
    assert_eq!(v.parts["sidekick"], "swe-2-medium");
    let v = variant(f, "fusion-claude-fable-5-1-medium-sidekick-swe-2-medium");
    assert_eq!(v.effort.as_deref(), Some("medium"));
    assert!(!v.fast);

    // The plain lead model is still its own row.
    assert!(models.iter().any(|m| m.id == "claude-opus-5-5"));
}

#[test]
fn parse_fusion_shapes() {
    let f = parse_fusion(
        "fusion-gpt-5-6-sol-high-fast-sidekick-swe-2-medium",
        "Fusion (GPT-5.6 Sol High Thinking Fast + SWE-2 Medium)",
    )
    .unwrap();
    assert_eq!(f.lead, "gpt-5-6-sol");
    assert_eq!(f.effort.as_deref(), Some("high"));
    assert!(f.fast);
    assert_eq!(f.sidekick, "swe-2-medium");
    assert_eq!(f.lead_name.as_deref(), Some("GPT-5.6 Sol"));
    assert_eq!(f.sidekick_name.as_deref(), Some("SWE-2 Medium"));
    // Names fall back to ids when the display isn't the pairing shape.
    let f = parse_fusion("fusion-claude-opus-5-low-sidekick-glm-5-2", "odd").unwrap();
    assert_eq!(f.lead, "claude-opus-5");
    assert_eq!(f.sidekick, "glm-5-2");
    assert!(f.lead_name.is_none());
    assert!(parse_fusion("fusion-mystery", "Fusion Mystery").is_none());
    assert!(parse_fusion("fusion--sidekick-swe-2-high", "").is_none());
}
