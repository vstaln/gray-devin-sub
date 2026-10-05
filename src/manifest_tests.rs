use super::*;

#[test]
fn manifest_ids_and_protocol() {
    let m = serde_json::to_value(manifest()).unwrap();
    assert_eq!(m["name"], "devin-sub");
    assert_eq!(m["version"], "0.1.0");
    assert_eq!(m["protocol"], "1.2");
    let caps: Vec<&str> = m["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert!(caps.contains(&"provider.credentials"));
    assert!(caps.contains(&"host.say"));

    let p = &m["providers"][0];
    assert_eq!(p["id"], "devin-subscription");
    assert_eq!(p["name"], "Devin subscription");
    let am = &p["auth_methods"][0];
    assert_eq!(am["id"], "devin-login");
    assert_eq!(am["kind"], "api_key");
    assert_eq!(am["name"], "Devin CLI login");
    let ops: Vec<&str> = am["operations"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| o.as_str())
        .collect();
    assert_eq!(ops, ["models", "chat"]);

    let t = &p["transport"];
    assert_eq!(t["kind"], "openai-responses");
    assert_eq!(t["base_url"], "https://127.0.0.1:1/");
    assert_eq!(t["authorization"]["kind"], "bearer");
    assert_eq!(t["authorization"]["secret_name"], "relay_token");
    let h = &t["headers"][0];
    assert_eq!(h["name"], "session-id");
    assert_eq!(h["source"]["kind"], "session_id");
    assert_eq!(h["required"], true);
    provider()
        .validate()
        .expect("subscription declaration validates");
}
