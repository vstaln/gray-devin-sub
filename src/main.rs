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

use devin_sub::{catalog, chat, manifest, models, relay, session, setup, usage};

use gray_plugin::ProviderRpcError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Write;

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

fn cli_help() -> String {
    "devin-sub — Devin subscription provider for gray\n\
     usage: devin-sub manifest | sidecar | models | tools [all|bash|+name|-name|name …]\n\
     slash:  /devin · /devin tools … · /fusion"
        .to_string()
}

fn main() -> anyhow::Result<()> {
    // `gray install plugin` registers sidecars by running `<bin> manifest`.
    match std::env::args_os().nth(1).as_deref() {
        Some(m) if m == "manifest" => {
            println!("{}", serde_json::to_string(&manifest::manifest())?);
            return Ok(());
        }
        Some(m) if m == "models" => {
            let catalog = models::catalog().map_err(anyhow::Error::msg)?;
            println!("{}", serde_json::to_string_pretty(&catalog)?);
            return Ok(());
        }
        // `gray devin-sub tools …` and the REPL's captured `/devin tools
        // …` both land here: one allowlist file, no sidecar needed.
        Some(t) if t == "tools" => {
            let args: Vec<String> = std::env::args_os()
                .skip(2)
                .map(|s| s.to_string_lossy().into_owned())
                .collect();
            println!("{}", devin_sub::settings::tools_command(&args));
            return Ok(());
        }
        Some(s) if s == "help" || s == "--help" || s == "-h" => {
            println!("{}", cli_help());
            return Ok(());
        }
        Some(s) if s == "sidecar" => {}
        Some(other) => {
            eprintln!("unknown verb {other:?}\n\n{}", cli_help());
            std::process::exit(2);
        }
        None if std::io::IsTerminal::is_terminal(&std::io::stdin()) => {
            println!("{}", cli_help());
            return Ok(());
        }
        None => {}
    }
    // Drop persisted pool records too old for any reattach to pay off.
    devin_sub::session::prune_persisted();
    // Sweep orphaned `/tmp/devin-sub-cfg-*` dirs left by sidecars that
    // died ungracefully (TempDir drop never ran). `session::IDLE_TTL`
    // reaps every pooled child, so a dir older than twice that cannot
    // belong to a live `devin acp` — delete-only-the-clearly-dead.
    std::thread::spawn(|| {
        let tmp = std::env::temp_dir();
        let cutoff = std::time::SystemTime::now()
            - (devin_sub::session::IDLE_TTL * 2 + std::time::Duration::from_secs(300));
        if let Ok(rd) = std::fs::read_dir(&tmp) {
            for e in rd.flatten() {
                let name = e.file_name();
                if !name.to_string_lossy().starts_with("devin-sub-cfg-") {
                    continue;
                }
                let stale = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .map(|m| m < cutoff)
                    .unwrap_or(false);
                if stale {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
    });

    // Re-pool sessions persisted before this restart while a live owner
    // would still be keepalive-warming them — their ~5min upstream cache
    // entries lapse otherwise. Best-effort, background.
    std::thread::spawn(devin_sub::session::rewarm);

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in std::io::BufRead::lines(stdin.lock()) {
        let line = line?;
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
        let outcome = handle(&request);
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

fn handle(request: &Request) -> Result<Value, ProviderRpcError> {
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
        "provider/auth/refresh" | "provider/auth/revoke" => {
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
        "provider/chat" => chat_turn(params),
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
            if argv.first().map(String::as_str) == Some("status") {
                return Ok(json!({"text": devin_sub::status::text()}));
            }
            manifest::run_command(name, &argv)
                .ok_or_else(|| ProviderRpcError::Protocol(format!("unknown command: {name}")))
        }
        "plugin/shutdown" => Ok(json!({})),
        other => Err(ProviderRpcError::Protocol(format!(
            "method not found: {other}"
        ))),
    }
}

/// One relayed turn: park the intent, open the loopback relay, and hand
/// the host the relay URL + per-turn bearer its standard Responses POST
/// uses. The admitted POST translates, spawns, folds and streams — the
/// sidecar answers from the relay, never from this method.
fn chat_turn(params: Value) -> Result<Value, ProviderRpcError> {
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
    let bearer = format!("devin-sub-{}", chat::rand_hex(16));
    let (port, _handle) = relay::start_turn_server(model.clone(), bearer.clone())
        .map_err(ProviderRpcError::Unavailable)?;
    std::mem::forget(_handle);
    Ok(json!({
        "relay_url": format!("http://127.0.0.1:{port}/relay/{bearer}/responses"),
        "relay_token": bearer,
        "native_model": catalog::native_model(&model, None),
    }))
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
