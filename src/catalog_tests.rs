use super::*;

const SAMPLE: &str = "SWE-2 (swe-2)\n  aliases: swe\n  swe-2-high                                                               SWE-2 High  [262K context, Free]\n  claude-opus-5-5-medium                                                   Claude Opus 5.5 Medium  [1M context, $4 / 1M Input · $30 / 1M Output]\n  swe-1-7-lightning                                                        SWE-1.7 Lightning Max  [202752 context, $2.5 / 1M Input]\n  adaptive                                                                 Adaptive  [$0.5 / 1M Input · $2.5 / 1M Output]\n";

#[test]
fn parses_model_rows() {
    let rows = parse_list(SAMPLE);
    let get = |id: &str| rows.iter().find(|r| r.id == id);
    assert!(get("swe-2-high").is_some());
    assert_eq!(get("swe-2-high").unwrap().context, Some(262_000));
    assert_eq!(get("claude-opus-5-5-medium").unwrap().context, Some(1_000_000));
    assert_eq!(
        get("claude-opus-5-5-medium").unwrap().display,
        "Claude Opus 5.5 Medium"
    );
    // Bare integer window (no K/M suffix).
    assert_eq!(get("swe-1-7-lightning").unwrap().context, Some(202_752));
    // No context bracket → None, never a guess.
    assert_eq!(get("adaptive").unwrap().context, None);
    // Family header skipped, alias exposed as an id with unknown window.
    assert!(get("SWE-2 (swe-2)").is_none());
    assert_eq!(get("swe").unwrap().context, None);
}

#[test]
fn skips_headers_and_blank() {
    let rows = parse_list("\nFamily (x)\n\n  only-id\n");
    // A row with no display/extra still parses (id only).
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "only-id");
}

#[test]
fn ids_pass_through_verbatim() {
    assert_eq!(native_model("gpt-6-sol-low"), "gpt-6-sol-low");
    assert_eq!(
        native_model("claude-opus-5-5-low-fast"),
        "claude-opus-5-5-low-fast"
    );
}
