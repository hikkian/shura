# Security

## Threat model

Shura is built for a single user on a single machine. Nothing in it is designed to be reachable from
a network.

| Component | Listens on | Authentication |
|---|---|---|
| AI Gateway | `127.0.0.1:8080` | none |
| llama-server (spawned by the gateway) | `127.0.0.1:8090` | none (optional `apiKey` in `model-launch.json`) |
| SearXNG container | `127.0.0.1:8888` | none |

**Do not expose these ports** (no `0.0.0.0`, no port forwarding, no reverse proxy to the internet).
Anyone who can reach the gateway can run the model, unload it, or disable it through `/guardian/*`.
If you need remote access, put it behind an authenticated tunnel such as SSH port forwarding or
Tailscale, and set `apiKey`.

## The agent runs commands on your machine

OpenCode executes shell commands and edits files with your user's permissions. The shipped
`opencode.json` asks before any shell command that is not on an allowlist, and denies destructive
ones (`rm -rf`, `mkfs`, `dd`, `shutdown`, …). Review that list before relaxing it. Content from web
pages, search results and MCP tools can contain prompt-injection attempts; treat an agent that has
read untrusted content with the same care as untrusted code.

## What the project stores locally

- `~/.local/state/ai-gateway/slots/`: at most one KV-cache file per mode (text/vision). It contains the
  model's internal state for your last long session, which can encode its content. Delete the folder
  if that matters to you.
- `config/guardian.json`, `config/model-launch.json`: machine-local paths. They are gitignored.
- OpenCode's session history (managed by OpenCode, in `~/.local/share/opencode/`).

No telemetry is sent by any component in this repository. OpenCode's own network calls are disabled
by the shipped config and environment (see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)).

## Reporting a vulnerability

Open a private security advisory on this repository (Security → Advisories → Report a vulnerability).
Please do not file public issues for security problems.
