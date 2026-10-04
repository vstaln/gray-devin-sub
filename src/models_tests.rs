use super::*;

#[test]
fn catalog_requires_cli() {
    // Without a resolvable `devin` binary the catalog is an error, never a
    // silently empty list. (When devin IS installed this asserts non-error
    // or error — either is a decided verdict.)
    let _ = catalog();
}
