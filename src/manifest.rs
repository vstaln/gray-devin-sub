//! The exact protocol-1.2 Devin subscription provider declaration.

use gray_plugin::{
    AuthMethodDecl, PROVIDER_CREDENTIALS, ProviderDecl, ProviderHeaderDecl,
    ProviderRequestPolicyDecl, ProviderTransportDecl,
};

pub const PLUGIN_NAME: &str = "devin-sub";
pub const PLUGIN_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PROVIDER_ID: &str = "devin-subscription";
pub const AUTH_METHOD_ID: &str = "devin-login";
/// Opens the host's `/model` picker focused on the folded Fusion row.
pub const FUSION_COMMAND: &str = "/fusion";

/// A protocol-1.2 manifest value. Credentials stay with the user's own
/// `devin` CLI login: the `devin-login` method performs no credential
/// handling, it only names the login the chat path probes before spawning.
pub fn manifest() -> gray_plugin::Manifest {
    gray_plugin::Manifest {
        name: PLUGIN_NAME.to_string(),
        version: PLUGIN_VERSION.to_string(),
        tools: Vec::new(),
        commands: vec!["/devin".to_string(), FUSION_COMMAND.to_string()],
        hooks: Vec::new(),
        protocol: Some("1.2".to_string()),
        subcommands: Vec::new(),
        capabilities: vec![
            PROVIDER_CREDENTIALS.to_string(),
            gray_plugin::capabilities::HOST_SAY.to_string(),
        ],
        providers: vec![provider()],
        provider_errors: Vec::new(),
    }
}

/// `command/run` result for a claimed command, `None` for names this
/// sidecar doesn't answer. `/fusion` asks the host to open its model
/// picker on the `fusion` row (`model_picker`); `text` is the fallback a
/// host without `model_picker` support prints instead.
pub fn run_command(name: &str) -> Option<serde_json::Value> {
    (name == FUSION_COMMAND).then(|| {
        serde_json::json!({
            "model_picker": crate::models::FUSION_ID,
            "text": "Open /model and pick Fusion (this gray is too old for /fusion)",
        })
    })
}

/// Devin subscription provider. Requests go to the loopback relay the
/// sidecar opens per chat turn (see `chat`); the host adds bearer, policy,
/// and every declared header from this declaration. The bearer is a
/// per-turn relay token minted by the sidecar, never the user's credential.
pub fn provider() -> ProviderDecl {
    ProviderDecl {
        id: PROVIDER_ID.to_string(),
        name: "Devin subscription".to_string(),
        transport: ProviderTransportDecl {
            kind: "openai-responses".to_string(),
            base_url: "https://127.0.0.1:1/"
                .parse()
                .expect("loopback placeholder"),
            authorization: gray_plugin::ProviderAuthorizationDecl {
                kind: "bearer".to_string(),
                secret_name: "relay_token".to_string(),
            },
            request: ProviderRequestPolicyDecl {
                prompt_cache_key: false,
                warm_replay: false,
                store: false,
                include_reasoning_encrypted: true,
                previous_response_id: false,
                tool_choice: Some("auto".to_string()),
                parallel_tool_calls: Some(true),
                text_verbosity: Some("low".to_string()),
                // Devin's per-ACP-session prompt cache runs ~5min
                // (measured; see session.rs) — tell the host's warmth
                // timer and cold-cache notices the real lifetime.
                cache_ttl_secs: Some(300),
            },
            headers: vec![ProviderHeaderDecl {
                name: "session-id".to_string(),
                value: None,
                source: Some(gray_plugin::ProviderHeaderSourceDecl::SessionId),
                required: true,
            }],
        },
        auth_methods: vec![AuthMethodDecl {
            id: AUTH_METHOD_ID.to_string(),
            name: "Devin CLI login".to_string(),
            kind: "api_key".to_string(),
            operations: vec!["models".to_string(), "chat".to_string()],
        }],
    }
}

#[path = "manifest_tests.rs"]
#[cfg(test)]
mod tests;
