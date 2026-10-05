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
fn families_share_a_tier_base() {
    let rows = parse_list(SAMPLE);
    let fams = families(&rows);
    let swe = fams.iter().find(|f| f.slug == "swe-2").unwrap();
    assert_eq!(swe.base, "swe-2");
    let variants: Vec<&str> = swe.members.iter().map(|(_, v)| *v).collect();
    assert_eq!(variants, vec!["high", "max"]);
    // The family base comes from member ids (dashed), not the header slug
    // (dotted): `claude-opus-5-5`, not `claude-opus-5.5`.
    let claude = fams.iter().find(|f| f.slug == "claude-opus-5.5").unwrap();
    assert_eq!(claude.base, "claude-opus-5-5");
    // A family whose member IS the base has a bare row → no collapse.
    let ids: std::collections::HashSet<&str> = rows.iter().map(|e| e.id.as_str()).collect();
    let adaptive = fams.iter().find(|f| f.slug == "adaptive").unwrap();
    assert_eq!(adaptive.base, "adaptive");
    assert!(family_efforts(adaptive, &ids).is_none());
    let lightning = fams.iter().find(|f| f.slug == "swe-1.7-lightning").unwrap();
    assert!(family_efforts(lightning, &ids).is_none());
    // Declared efforts come back in picker order.
    assert_eq!(
        family_efforts(swe, &ids).unwrap(),
        vec!["high".to_string(), "max".to_string()]
    );
}

#[test]
fn ids_pass_through_verbatim() {
    assert_eq!(native_model("gpt-6-sol-low", None), "gpt-6-sol-low");
    assert_eq!(
        native_model("claude-opus-5-5-low-fast", Some("max")),
        "claude-opus-5-5-low-fast"
    );
}
