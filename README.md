# devin-sub — Devin subscription provider sidecar for Gray

A protocol-1.2 provider sidecar that puts a **Devin subscription** behind
Gray's standard OpenAI-Responses transport: Gray keeps host-owned tools,
approvals, and compaction; Devin's CLI answers model requests.

## Why this is ToS-safe

The sidecar talks to Devin exclusively through the **official
`devin acp` command** — Devin's documented Agent Client Protocol
integration, the same path Zed, JetBrains, and `hermes-devin-acp` use.
There is no token handling anywhere: authentication lives entirely inside
the Devin CLI (`devin auth login`); the sidecar never reads credentials
files, never captures tokens, and refuses to run when a conflicting
`DEVIN_*` environment variable would reroute the login.

## How a turn works

1. `provider/chat` opens a per-turn loopback relay and hands the host a
   one-shot bearer URL. The relay admits exactly one POST — the
   "one upstream request per chat turn" budget is enforced by a
   consumed-token admission guard, not by trust.
2. The admitted OpenAI Responses body becomes ONE `devin acp` session
   (`initialize → session/new → session/prompt → session/delete`) in a
   scratch cwd whose `.devin/config.json` denies every native tool, and
   under a generated user config with subagents/MCP/auto-update off.
3. Gray's tools ride in as a text funnel contract (`​```gray_calls`
   fenced block); models that ignore it and reach for a native `exec` /
   `read` are caught by `tool_call` notifications and redirected onto the
   host's `bash` tool, then the session is cancelled.
4. The reply (text, thought summary, calls, usage) is folded back into
   the Responses SSE stream the host already understands.

`session/request_permission` is always answered `cancelled` — a harness
turn never waits on an interactive prompt.

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
