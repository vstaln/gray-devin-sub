use super::*;

const SAMPLE: &str = "SWE-2 (swe-2)\n  aliases: swe\n  swe-2-high                                                               SWE-2 High  [262K context, Free]\n  swe-2-max                                                                SWE-2 Max  [262K context, Free]\nClaude Opus 5.5 (claude-opus-5.5)\n  claude-opus-5-5-medium                                                   Claude Opus 5.5 Medium  [1M context, $4 / 1M Input · $30 / 1M Output]\nSWE-1.7 Lightning (swe-1.7-lightning)\n  swe-1-7-lightning                                                        SWE-1.7 Lightning Max  [202752 context, $2.5 / 1M Input]\nAdaptive (adaptive)\n  adaptive                                                                 Adaptive  [$0.5 / 1M Input · $2.5 / 1M Output]\n";

#[test]
fn parses_model_rows() {
    let rows = parse_list(SAMPLE);
    let get = |id: &str| rows.iter().find(|r| r.id == id);
    assert!(get("swe-2-high").is_some());
    assert_eq!(get("swe-2-high").unwrap().context, Some(262_000));
    assert_eq!(get("swe-2-high").unwrap().family.slug, "swe-2");
    assert!(!get("swe-2-high").unwrap().alias);
    assert_eq!(
        get("claude-opus-5-5-medium").unwrap().context,
        Some(1_000_000)
    );
    assert_eq!(
        get("claude-opus-5-5-medium").unwrap().display,
        "Claude Opus 5.5 Medium"
    );
    assert_eq!(
        get("claude-opus-5-5-medium").unwrap().family.slug,
        "claude-opus-5.5"
    );
    // Bare integer window (no K/M suffix).
    assert_eq!(get("swe-1-7-lightning").unwrap().context, Some(202_752));
    // No context bracket → None, never a guess.
    assert_eq!(get("adaptive").unwrap().context, None);
    // Family header is not itself a row; the alias is, flagged as one.
    assert!(get("SWE-2 (swe-2)").is_none());
    assert_eq!(get("swe").unwrap().context, None);
    assert!(get("swe").unwrap().alias);
    assert_eq!(get("swe").unwrap().family.slug, "swe-2");
}

#[test]
fn skips_headers_and_blank() {
    let rows = parse_list("\nFamily (x)\n\n  only-id  Only Id\n");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "only-id");
    assert_eq!(rows[0].family.slug, "x");
}

#[test]
fn plain_groups_share_a_tier_base() {
    let rows = parse_list(SAMPLE);
    let groups = plain_groups(&rows);
    let swe = groups.iter().find(|g| g.base == "swe-2").unwrap();
    let efforts: Vec<Option<&str>> = swe.members.iter().map(|(_, e, _)| e.as_deref()).collect();
    assert_eq!(efforts, vec![Some("high"), Some("max")]);
    assert_eq!(swe.slug(), Some("swe-2"));
    assert_eq!(swe.name(), "SWE-2");
    // The base comes from member ids (dashed), not the header slug
    // (dotted): `claude-opus-5-5`, not `claude-opus-5.5`.
    assert!(groups.iter().any(|g| g.base == "claude-opus-5-5"));
    // A bare id whose display names a tier is pinned to it.
    let lightning = groups
        .iter()
        .find(|g| g.base == "swe-1-7-lightning")
        .unwrap();
    assert_eq!(lightning.members[0].1.as_deref(), Some("max"));
    // A lone tier-less id has nothing to fold.
    assert!(groups.iter().all(|g| g.base != "adaptive"));
    // Declared efforts come back in picker order.
    assert_eq!(swe.efforts(), vec!["high".to_string(), "max".to_string()]);
}

#[test]
fn parse_plain_requires_display_confirmation() {
    let p = parse_plain("gpt-6-sol-none-priority", "GPT-6 Sol No Thinking Fast").unwrap();
    assert_eq!(p.base, "gpt-6-sol");
    assert_eq!(p.effort.as_deref(), Some("off"));
    assert!(p.fast);
    let p = parse_plain("claude-opus-5-5-high-fast", "Claude Opus 5.5 High Fast").unwrap();
    assert_eq!(p.base, "claude-opus-5-5");
    assert_eq!(p.effort.as_deref(), Some("high"));
    assert!(p.fast);
    let p = parse_plain("inkling-xhigh", "Inkling X-High").unwrap();
    assert_eq!(p.effort.as_deref(), Some("xhigh"));
    // A tier-looking suffix the display doesn't confirm stays in the base.
    let p = parse_plain("widget-max", "Widget").unwrap();
    assert_eq!(p.base, "widget-max");
    assert_eq!(p.effort, None);
    let p = parse_plain("widget-fast", "Widget").unwrap();
    assert_eq!(p.base, "widget-fast");
    assert!(!p.fast);
    // Context and thinking products keep their own base.
    assert_eq!(
        parse_plain("glm-5-2-max-1m", "GLM-5.2 Max 1M")
            .unwrap()
            .base,
        "glm-5-2-max-1m"
    );
    assert_eq!(
        parse_plain("claude-opus-4-6-thinking", "Claude Opus 4.6 Thinking")
            .unwrap()
            .base,
        "claude-opus-4-6-thinking"
    );
    // Legacy and fusion ids are not plain.
    assert!(parse_plain("MODEL_GPT_5_2_LOW", "GPT-5.2 Low Thinking").is_none());
    assert!(
        parse_plain(
            "fusion-claude-opus-5-5-high-sidekick-swe-2-high",
            "Fusion (Claude Opus 5.5 High + SWE-2 High)"
        )
        .is_none()
    );
}

#[test]
fn display_words() {
    assert_eq!(display_tier("GPT-6 Sol High Thinking Fast"), Some("high"));
    assert_eq!(display_tier("GPT-5.4 No Thinking"), Some("none"));
    assert_eq!(display_tier("Nemotron 3 Ultra None"), Some("none"));
    assert_eq!(display_tier("SWE-1.6 Fast"), None);
    assert_eq!(base_display("GPT-6 Sol High Thinking Fast"), "GPT-6 Sol");
    assert_eq!(base_display("GPT-6 Sol No Thinking"), "GPT-6 Sol");
    assert_eq!(base_display("SWE-1.6 Fast"), "SWE-1.6");
    assert_eq!(base_display("Claude Fable 5.1 Medium"), "Claude Fable 5.1");
}

#[test]
fn legacy_family_ids_resolve_by_effort() {
    let rows = parse_list(SAMPLE);
    // Old configs saved the folded family id; effort picks the tier.
    assert_eq!(resolve_native(&rows, "swe-2", Some("max")), "swe-2-max");
    assert_eq!(resolve_native(&rows, "swe-2", Some("high")), "swe-2-high");
    // No matching tier (or none asked) → the routable header slug.
    assert_eq!(resolve_native(&rows, "swe-2", Some("low")), "swe-2");
    assert_eq!(
        resolve_native(&rows, "claude-opus-5-5", None),
        "claude-opus-5.5"
    );
    // Real ids — variant picks — pass through whatever the effort says.
    assert_eq!(
        resolve_native(&rows, "swe-2-high", Some("max")),
        "swe-2-high"
    );
    assert_eq!(
        resolve_native(&rows, "swe-1-7-lightning", Some("medium")),
        "swe-1-7-lightning"
    );
    // Unknown ids (and the Fusion header slug) pass through.
    assert_eq!(resolve_native(&rows, "fusion", Some("high")), "fusion");
    assert_eq!(resolve_native(&rows, "nope", None), "nope");
}

#[test]
fn ids_pass_through_verbatim() {
    assert_eq!(native_model("gpt-6-sol-low", None), "gpt-6-sol-low");
    assert_eq!(
        native_model("claude-opus-5-5-low-fast", Some("max")),
        "claude-opus-5-5-low-fast"
    );
}
