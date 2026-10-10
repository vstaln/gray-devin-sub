//! devin-sub: a protocol-1.2 provider sidecar for Gray.
//!
//! Devin subscription through the official `devin acp` CLI: the host
//! owns tools, approvals and compaction, the sidecar only answers — one
//! upstream request per chat turn, enforced by the admission relay.
//!
//! Wire (NDJSON over stdio, host ids are opaque):
//! - `plugin/manifest` → manifest + the `devin-subscription` provider decl.
//! - `provider/models` → live `devin models list` catalog.
//! - `provider/chat` → one relayed turn (declares the per-turn relay URL +
//!   bearer the host's standard Responses POST must use).
//! - `provider/auth/*` → the user's own `devin auth login` owns
//!   credentials; start/poll report external-login status, refresh/revoke
//!   are unsupported.
//! - `command/run` → `/fusion` opens the host's model picker on the
//!   folded Fusion row (`{"model_picker":"fusion","text":<fallback>});
//!   a bare `/devin` answers nothing so the host's provider-login
//!   shortcut switches the session to Devin, and `/devin tools …` owns
//!   the upstream tool allowlist (`{"text": …}`).
//! - `plugin/shutdown` → clean exit. Unknown methods are protocol errors
//!   (provider sidecars must fail loudly, never hang a turn).

use devin_sub::{catalog, manifest, models, relay, session, setup, usage};

use gray_plugin::{ProviderRefreshRequest, ProviderRevokeRequest, ProviderRpcError};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Serialize)]
struct Response {
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

/// Active relay intents by per-turn token (see `relay::RelayIntent`):
/// the relay server fills the transcript in when the host POSTs.
type Relays = relay::Intents;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    // `gray install plugin` registers sidecars by running `<bin> manifest`.
    match std::env::args().nth(1).as_deref() {
        Some("manifest") => {
            println!("{}", serde_json::to_string(&manifest::manifest())?);
            return Ok(());
        }
        Some("models") => {
            let catalog = models::catalog().map_err(anyhow::Error::msg)?;
            println!("{}", serde_json::to_string_pretty(&catalog)?);
            return Ok(());
        }
        // `gray devin-sub tools …` and the REPL's captured `/devin tools
        // …` both land here: one allowlist file, no sidecar needed.
        Some("tools") => {
            let args: Vec<String> = std::env::args().skip(2).collect();
            println!("{}", devin_sub::settings::tools_command(&args));
            return Ok(());
        }
        _ => {}
    }
    let relays: Relays = Arc::new(Mutex::new(HashMap::new()));
    let mut lines =
        tokio::io::AsyncBufReadExt::lines(tokio::io::BufReader::new(tokio::io::stdin()));
    let mut stdout = std::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(_) => continue,
        };
        if request.id.is_null() {
            if request.method == "plugin/shutdown" {
                session::shutdown();
                return Ok(());
            }
            continue;
        }
        let outcome = handle(&relays, &request).await;
        let response = match outcome {
            Ok(result) => Response {
                id: request.id,
                result: Some(result),
                error: None,
            },
            Err(error) => Response {
                id: request.id,
                result: None,
                error: Some(error_value(error)),
            },
        };
        let frame = serde_json::to_string(&response)?;
        // Protocol frames only: never log credential payloads or upstream bodies.
        let _ = writeln!(stdout, "{frame}");
        stdout.flush()?;
        if request.method == "plugin/shutdown" {
            session::shutdown();
            return Ok(());
        }
    }
    // The host's stdin went away: close pooled sessions and the shared
    // stage before exiting.
    session::shutdown();
    Ok(())
}

async fn handle(relays: &Relays, request: &Request) -> Result<Value, ProviderRpcError> {
    let params = request.params.clone().unwrap_or_else(|| json!({}));
    match request.method.as_str() {
        "plugin/manifest" => Ok(serde_json::to_value(manifest::manifest()).unwrap()),
        "provider/auth/start" => {
            ensure_provider(
                params
                    .get("provider")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                params
                    .get("auth_method")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )?;
            // External login: the CLI owns credentials. `devin auth status`
            // answers in seconds and exits 0 even when logged out, so the
            // verdict comes from its (captured, never echoed) text.
            if setup::resolve_command().is_none() {
                return Err(ProviderRpcError::Unavailable(setup::INSTALL_HINT.into()));
            }
            match setup::probe_login() {
                setup::LoginState::LoggedIn => Ok(json!({"status": "authenticated"})),
                setup::LoginState::LoggedOut | setup::LoginState::Unknown => {
                    Err(ProviderRpcError::Unavailable(setup::LOGIN_HINT.into()))
                }
            }
        }
        "provider/auth/poll" => Err(ProviderRpcError::Protocol(
            "external login has no pollable operation".into(),
        )),
        "provider/auth/cancel" => Ok(json!({})),
        "provider/auth/refresh" => {
            let req: ProviderRefreshRequest =
                serde_json::from_value(params).map_err(|_| invalid("provider refresh request"))?;
            ensure_provider(&req.provider, &req.auth_method)?;
            Ok(json!({"status": "unsupported"}))
        }
        "provider/auth/revoke" => {
            let req: ProviderRevokeRequest =
                serde_json::from_value(params).map_err(|_| invalid("provider revoke request"))?;
            ensure_provider(&req.provider, &req.auth_method)?;
            Ok(json!({"status": "unsupported"}))
        }
        "provider/models" => {
            let provider = params
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let auth_method = params
                .get("auth_method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            ensure_provider(provider, auth_method)?;
            let catalog = models::catalog().map_err(ProviderRpcError::Unavailable)?;
            Ok(serde_json::to_value(catalog).unwrap())
        }
        "provider/chat" => chat_turn(relays, params).await,
        "provider/usage" => {
            let provider = params
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let auth_method = params
                .get("auth_method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            ensure_provider(provider, auth_method)?;
            usage::handle()
        }
        "command/run" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let argv: Vec<String> = params
                .get("argv")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            manifest::run_command(name, &argv)
                .ok_or_else(|| ProviderRpcError::Protocol("unknown command".into()))
        }
        "plugin/shutdown" => Ok(json!({})),
        _ => Err(ProviderRpcError::Protocol("unknown provider method".into())),
    }
}

/// One relayed turn: park the intent, open the loopback relay, and hand
/// the host the relay URL + per-turn bearer its standard Responses POST
/// uses. The admitted POST translates, spawns, folds and streams — the
/// sidecar answers from the relay, never from this method.
async fn chat_turn(relays: &Relays, params: Value) -> Result<Value, ProviderRpcError> {
    let provider = params
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if provider != manifest::PROVIDER_ID {
        return Err(ProviderRpcError::Protocol("unknown provider".into()));
    }
    let model = params
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("swe-2-medium")
        .to_string();
    // Fail fast before opening anything: no binary, no login, no relay.
    if setup::resolve_command().is_none() {
        return Err(ProviderRpcError::Unavailable(setup::INSTALL_HINT.into()));
    }
    // No login, no relay: `/connect` shows the hint instead of a turn that
    // dies later. A confirmed login is remembered for this sidecar's life;
    // an inconclusive probe never blocks a turn.
    static LOGGED_IN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED_IN.load(std::sync::atomic::Ordering::Relaxed) {
        match setup::probe_login() {
            setup::LoginState::LoggedOut => {
                return Err(ProviderRpcError::Unavailable(setup::LOGIN_HINT.into()));
            }
            setup::LoginState::LoggedIn => {
                LOGGED_IN.store(true, std::sync::atomic::Ordering::Relaxed)
            }
            setup::LoginState::Unknown => {}
        }
    }
    let bearer = format!("devin-sub-{}", hex_id());
    let intent = relay::RelayIntent {
        model: model.clone(),
    };
    relays
        .lock()
        .map(|mut r| {
            r.insert(bearer.clone(), intent);
        })
        .ok();
    let relays2 = relays.clone();
    let bearer2 = bearer.clone();
    let (port, _handle) =
        relay::start_turn_server(relays2, bearer2).map_err(ProviderRpcError::Unavailable)?;
    std::mem::forget(_handle);
    Ok(json!({
        "relay_url": format!("http://127.0.0.1:{port}/relay/{bearer}/responses"),
        "relay_token": bearer,
        "native_model": catalog::native_model(&model, None),
    }))
}

fn hex_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:08x}{:08x}", t, std::process::id())
}

fn ensure_provider(provider: &str, auth_method: &str) -> Result<(), ProviderRpcError> {
    if provider == manifest::PROVIDER_ID && auth_method == manifest::AUTH_METHOD_ID {
        Ok(())
    } else {
        Err(ProviderRpcError::Protocol(
            "unknown provider or auth method".into(),
        ))
    }
}

fn invalid(message: &'static str) -> ProviderRpcError {
    ProviderRpcError::Protocol(message.into())
}

fn error_value(error: ProviderRpcError) -> Value {
    match error {
        ProviderRpcError::Rpc(failure) => json!({
            "code": failure.code,
            "message": failure.message,
            "retryable": failure.retryable,
            "terminal": failure.terminal,
        }),
        ProviderRpcError::Protocol(message) => {
            json!({"code": "protocol", "message": message})
        }
        ProviderRpcError::Unavailable(message) => {
            json!({"code": "unavailable", "message": message})
        }
        ProviderRpcError::CapabilityMissing(message) => {
            json!({"code": "capability_missing", "message": message})
        }
    }
}
