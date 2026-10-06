//! Loopback admission relay: the single-request gate in front of
//! `devin acp`.
//!
//! Per chat turn the sidecar opens a loopback HTTP server and hands the
//! host a per-turn relay URL + bearer. The host POSTs its standard
//! Responses body there; the relay admits exactly ONE request (anything
//! past it gets 400 `ADMISSION_CONSUMED`), translates it to one funnel
//! turn, drives a pooled `devin acp` session (which makes exactly one
//! upstream request on its own), folds the native transcript to Responses
//! SSE and streams it back.
//!
//! A turn can outlive the host's per-read timeout: past `HEADER_GRACE`
//! the response becomes a close-delimited SSE stream kept alive with
//! `: keepalive` comments every `KEEPALIVE`. If the client goes away the
//! in-flight native turn is cancelled.
//!
//! Credentials are forwarded, never persisted: the bearer is a per-turn
//! token minted by the sidecar, never the user's OAuth token.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How long the relay may hold the response headers before committing to
/// the close-delimited keepalive path (the host's per-read timeout).
const HEADER_GRACE: Duration = Duration::from_secs(20);
/// SSE comment cadence once the response is streaming.
const KEEPALIVE: Duration = Duration::from_secs(15);

/// How often a waiting relay checks whether the client hung up.
const PEER_POLL: Duration = Duration::from_millis(250);

/// The client closed its end (gray interrupted the turn). Non-blocking
/// peek: EOF or a hard error means gone; no data yet means still there.
fn client_gone(s: &std::net::TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return false;
    }
    let gone = match s.peek(&mut [0u8; 1]) {
        Ok(n) => n == 0,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
    };
    let _ = s.set_nonblocking(false);
    gone
}

enum Wait {
    Done(Result<Vec<u8>, String>),
    Timeout,
    Died,
    Gone,
}

/// `recv_timeout` that also watches the peer: a hang-up sets `cancel` at
/// once, so the native turn stops instead of finishing for nobody.
fn wait_turn(
    rx: &mpsc::Receiver<Result<Vec<u8>, String>>,
    s: &std::net::TcpStream,
    cancel: &AtomicBool,
    span: Duration,
) -> Wait {
    let end = Instant::now() + span;
    loop {
        let step = end.saturating_duration_since(Instant::now()).min(PEER_POLL);
        match rx.recv_timeout(step) {
            Ok(r) => return Wait::Done(r),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Wait::Died,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if client_gone(s) {
                    cancel.store(true, Ordering::SeqCst);
                    return Wait::Gone;
                }
                if Instant::now() >= end {
                    return Wait::Timeout;
                }
            }
        }
    }
}

/// What `provider/chat` parked for one turn: the relay fills the body in.
pub struct RelayIntent {
    pub model: String,
}

pub type Intents = Arc<Mutex<HashMap<String, RelayIntent>>>;

/// Start the per-turn relay server. Returns the loopback port + handle.
/// The admitted POST does the whole turn synchronously; the server stays
/// open afterwards only to answer 400s to anything past the single
/// admission (native retries must see denial, not a hang).
pub fn start_turn_server(
    intents: Intents,
    bearer: String,
) -> Result<(u16, std::thread::JoinHandle<()>), String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("relay bind: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("relay addr: {e}"))?
        .port();
    let used = Arc::new(AtomicBool::new(false));
    let handle = std::thread::spawn(move || serve(listener, intents, bearer, used));
    Ok((port, handle))
}

const MAX_BODY: usize = 256 << 20;

fn serve(listener: TcpListener, intents: Intents, bearer: String, used: Arc<AtomicBool>) {
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        handle_conn(&mut s, &intents, &bearer, &used);
    }
}

fn handle_conn(
    s: &mut std::net::TcpStream,
    intents: &Intents,
    bearer: &str,
    used: &Arc<AtomicBool>,
) {
    let _ = s.set_read_timeout(Some(Duration::from_secs(330)));
    let mut buf = vec![0u8; 65536];
    let mut head = Vec::new();
    loop {
        let Ok(n) = s.read(&mut buf) else { return };
        if n == 0 {
            return;
        }
        head.extend_from_slice(&buf[..n]);
        if head.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if head.len() > 1_048_576 {
            return;
        }
    }
    let head_str = String::from_utf8_lossy(&head).into_owned();
    let mut lines = head_str.lines();
    let request_line = lines.next().unwrap_or("").to_string();
    let mut len = 0usize;
    let mut auth = String::new();
    for line in lines {
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let Some((name, v)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            len = v.trim().parse().unwrap_or(0);
        } else if name.eq_ignore_ascii_case("authorization") {
            auth = v.trim().to_string();
        }
    }
    let header_end = head
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(head.len());
    let mut body = head[header_end..].to_vec();
    // Image-heavy histories run past 8 MiB; read the whole declared body.
    while body.len() < len.min(MAX_BODY) {
        let Ok(n) = s.read(&mut buf) else { return };
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    let ok_path = request_line.starts_with("POST /relay/")
        && request_line.contains("/responses")
        && request_line.contains(bearer);
    let ok_bearer = auth == format!("Bearer {bearer}");
    if !ok_path || !ok_bearer {
        write_resp(s, 404, b"not found");
        return;
    }
    if len > MAX_BODY || body.len() < len {
        let body = json!({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": format!("relay body incomplete: read {} of {len} bytes (limit {MAX_BODY})", body.len()),
        }});
        write_resp(
            s,
            if len > MAX_BODY { 413 } else { 400 },
            body.to_string().as_bytes(),
        );
        return;
    }
    if used.swap(true, Ordering::SeqCst) {
        let body = json!({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": crate::chat::ADMISSION_CONSUMED,
        }});
        write_resp(s, 400, body.to_string().as_bytes());
        return;
    }
    // Admitted: the one request. The turn runs on a worker thread so the
    // connection can answer within HEADER_GRACE — or commit to a
    // close-delimited stream and keep it alive until the turn ends.
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>();
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let intents = intents.clone();
        let bearer = bearer.to_string();
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            let _ = tx.send(run_turn(&intents, &bearer, &body, &cancel));
        });
    }
    match wait_turn(&rx, s, &cancel, HEADER_GRACE) {
        Wait::Gone => {}
        Wait::Done(Ok(sse)) => {
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                sse.len()
            );
            let _ = s.write_all(header.as_bytes());
            let _ = s.write_all(&sse);
            let _ = s.flush();
        }
        Wait::Done(Err(detail)) => {
            let body = json!({"type": "error", "error": {
                "type": "server_error", "message": detail,
            }});
            write_resp(s, 500, body.to_string().as_bytes());
        }
        Wait::Died => {
            let body = json!({"type": "error", "error": {
                "type": "server_error", "message": "turn worker died",
            }});
            write_resp(s, 500, body.to_string().as_bytes());
        }
        Wait::Timeout => {
            // Slow turn: no Content-Length, the stream ends at close.
            // eventsource-stream ignores `:` comment lines.
            if s
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                )
                .and_then(|_| s.flush())
                .is_err()
            {
                cancel.store(true, Ordering::SeqCst);
                return;
            }
            loop {
                match wait_turn(&rx, s, &cancel, KEEPALIVE) {
                    Wait::Done(Ok(sse)) => {
                        let _ = s.write_all(&sse);
                        let _ = s.flush();
                        return;
                    }
                    Wait::Done(Err(detail)) => {
                        let failed = json!({"type": "response.failed",
                            "response": {"status": "failed",
                                "error": {"code": "server_error", "message": detail}}});
                        let _ =
                            s.write_all(format!("data: {failed}\n\ndata: [DONE]\n\n").as_bytes());
                        let _ = s.flush();
                        return;
                    }
                    Wait::Timeout => {
                        if s.write_all(b": keepalive\n\n")
                            .and_then(|_| s.flush())
                            .is_err()
                        {
                            cancel.store(true, Ordering::SeqCst);
                            return;
                        }
                    }
                    Wait::Gone | Wait::Died => return,
                }
            }
        }
    }
}

fn run_turn(
    intents: &Intents,
    bearer: &str,
    raw: &[u8],
    cancel: &Arc<AtomicBool>,
) -> Result<Vec<u8>, String> {
    let intent = intents
        .lock()
        .map(|mut m| m.remove(bearer))
        .ok()
        .flatten()
        .ok_or_else(|| "relay intent expired".to_string())?;
    let body: Value = serde_json::from_slice(raw)
        .map_err(|e| format!("relay body is not JSON ({} bytes): {e}", raw.len()))?;
    let turn = crate::chat::prepare_turn(&body, &intent.model)?;
    // A pooled session when this history continues one, else a fresh
    // spawn — either way native makes exactly one upstream request.
    let result = crate::chat::run_turn(&turn, cancel, Duration::from_secs(600))?;
    crate::chat::fold_result(&result)
}

fn write_resp(s: &mut std::net::TcpStream, status: u16, body: &[u8]) {
    let reason = match status {
        400 => "Bad Request",
        413 => "Payload Too Large",
        404 => "Not Found",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = s.write_all(header.as_bytes());
    let _ = s.write_all(body);
    let _ = s.flush();
}
