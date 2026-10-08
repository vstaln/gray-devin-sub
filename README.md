<p align="center">
  <img src="assets/gray-logo.svg" alt="gray" width="96">
</p>
<h1 align="center">gray-devin-sub</h1>
<p align="center">Run gray on a Devin subscription through the official <code>devin acp</code> CLI.</p>
<p align="center">
  <a href="https://github.com/vstaln/gray-devin-sub/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="gray plugin" src="https://img.shields.io/badge/gray-plugin-7aa2f7.svg">
  <img alt="rust" src="https://img.shields.io/badge/built%20with-rust-orange.svg">
</p>

A protocol-1.2 provider sidecar that puts a **Devin subscription** behind
Gray's standard OpenAI-Responses transport: Gray keeps host-owned tools,
approvals, and compaction; Devin's CLI answers model requests.

## Why this is ToS-safe

The sidecar talks to Devin exclusively through the **official
`devin acp` command** — Devin's documented Agent Client Protocol
integration, the same path editors like Zed and JetBrains use.
There is no token handling anywhere: authentication lives entirely inside
the Devin CLI (`devin auth login`); the sidecar never reads credentials
files, never captures tokens, and refuses to run when a conflicting
`DEVIN_*` environment variable would reroute the login.

## How a turn works

1. `provider/chat` opens a per-turn loopback relay and hands the host a
   one-shot bearer URL. The relay admits exactly one POST — the
   "one upstream request per chat turn" budget is enforced by a
   consumed-token admission guard, not by trust.
2. The admitted OpenAI Responses body is answered by a **pooled**
   `devin acp` session (`initialize → session/new`, then one
   `session/prompt` per turn). Children run in a shared scratch cwd whose
   `.devin/config.json` denies every native tool, under a generated user
   config with subagents/MCP/auto-update off. Devin's prompt cache is
   per-session: when the request's history strictly continues what a
   pooled session last answered — same model, same prepared system text,
   same prefix, and the host's replay of the last answer byte-identical —
   that session gets prompted with only the delta (`[Tool result]` /
   `[User]` blocks) and ~the whole transcript is already upstream.
   Anything else spawns a fresh `devin acp`. Sessions idle 15 min (or
   dead) are reaped; the pool caps at 4 children.
3. Gray's tools ride in as a text funnel contract (`​```gray_calls`
   fenced block); models that ignore it and reach for a native `exec` /
   `read` are caught by `tool_call` notifications and redirected onto the
   host's `bash` tool, then the prompt is cancelled.
4. The reply (text, thought summary, calls, usage) is folded back into
   the Responses SSE stream the host already understands. Turns that run
   past 20s answer as a close-delimited stream with `: keepalive`
   comments every 15s — the host's HTTP client times out idle reads — and
   a client disconnect sends `session/cancel` upstream instead of letting
   an abandoned turn bill.

`session/request_permission` is always answered `cancelled` — a harness
turn never waits on an interactive prompt.

Set `DEVIN_SUB_DEBUG=1` to append one line per turn (reuse/fresh reason,
ACP session id, prompt chars, usage) to
`$TMPDIR/devin-sub-<pid>.log` — no prompt content.

## Build & install

```sh
cargo build --release
install -m755 target/release/devin-sub ~/.local/bin/gray-devin-sub
GRAY_PLUGIN_PATH=$HOME/.local/bin/gray-devin-sub gray install plugin devin-sub
```

Then reference the provider as `devin-sub:devin-subscription` in your
Gray config (see `gray-antigravity-sub`'s README for the full
config shape — this sidecar is a drop-in sibling).

Requires the Devin CLI on `PATH` and `devin auth login` already done.
Override the binary with `GRAY_DEVIN_SUB_COMMAND` if needed.

---
Part of the [gray](https://github.com/vstaln/gray) plugin ecosystem —
the open-source AI agent harness. <https://gray.alignment.id>
