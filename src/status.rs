//! `/devin status` — one glanceable panel: `devin auth status` identity +
//! tier, then `usage::handle()`'s ACU windows. Same convention `nous-sub
//! status` set: login / acct / plan / windows.

use serde_json::Value;

use crate::{setup, usage};

/// "  Name:  X" / "  Email:  X" / "  Tier:  X" from `devin auth status`.
fn field(output: &str, name: &str) -> Option<String> {
    for line in output.lines() {
        if let Some(v) = line.trim().strip_prefix(&format!("{name}:")) {
            let v = v.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn window_line(w: &Value) -> Option<String> {
    let label = w.get("label")?.as_str()?;
    let mut line = format!("  {label}:");
    if let Some(p) = w.get("used_percent").and_then(Value::as_f64) {
        line.push_str(&format!(" {p:.0}% used"));
    } else if let (Some(u), Some(l), Some(unit)) = (
        w.get("used").and_then(Value::as_f64),
        w.get("limit").and_then(Value::as_f64),
        w.get("unit").and_then(Value::as_str),
    ) {
        line.push_str(&format!(" {u:.1}/{l:.1} {unit}"));
    } else if let Some(l) = w.get("limit").and_then(Value::as_f64) {
        let unit = w.get("unit").and_then(Value::as_str).unwrap_or("");
        line.push_str(&format!(" limit {l:.0} {unit}"));
    }
    if let Some(r) = w.get("resets_at").and_then(Value::as_str) {
        line.push_str(&format!(" · resets {r}"));
    }
    Some(line)
}

pub fn text() -> String {
    let mut out = "Devin backend (unofficial)".to_string();
    let (state, output) = setup::auth_status_output();
    match state {
        setup::LoginState::LoggedIn => {
            out.push_str("\n  login: devin auth (devin CLI) · logged in")
        }
        setup::LoginState::LoggedOut => {
            out.push_str("\n  login: logged out — run `devin auth login`")
        }
        setup::LoginState::Unknown => out.push_str("\n  login: unknown (probe inconclusive)"),
    }
    if let Some(text) = output.as_deref() {
        let name = field(text, "Name");
        let email = field(text, "Email");
        let acct = match (name, email) {
            (Some(n), Some(e)) => Some(format!("{n} <{e}>")),
            (None, Some(e)) => Some(e),
            (Some(n), None) => Some(n),
            (None, None) => None,
        };
        if let Some(acct) = acct {
            out.push_str(&format!("\n  acct:  {acct}"));
        }
        if let Some(tier) = field(text, "Tier") {
            out.push_str(&format!("\n  tier:  {tier}"));
        }
    }
    match usage::handle() {
        Ok(v) => {
            if let Some(plan) = v.get("plan").and_then(Value::as_str)
                && plan != "unknown"
            {
                out.push_str(&format!("\n  plan:  {plan}"));
            }
            if let Some(wins) = v.get("windows").and_then(Value::as_array) {
                for w in wins {
                    if let Some(line) = window_line(w) {
                        out.push('\n');
                        out.push_str(&line);
                    }
                }
            }
            if let Some(note) = v.get("note").and_then(Value::as_str)
                && !note.is_empty()
            {
                out.push_str(&format!("\n  note:  {note}"));
            }
        }
        Err(e) => out.push_str(&format!("\n  usage: ({e})")),
    }
    out
}
