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

### Web pages in your browser

A page open in your browser can send requests to `127.0.0.1` too, so the gateway does not trust a request just because it
comes from this machine. It refuses what a browser can send for a stranger:

- a `Host` header that is not this machine (a DNS-rebinding page uses its own name);
- an `Origin` that is not the gateway's own page. To let a web UI of yours call the gateway, list its origin in
  `allowedOrigins` in `guardian.json` (for example `["http://localhost:3000"]`); the gateway then answers that origin only,
  never `*`;
- a request the browser marks as cross-site (`Sec-Fetch-Site`), which also covers an `<img>` tag;
- a GET on anything under `/guardian/` that changes something: those endpoints take POST only.

llama-server's own CORS headers (it answers every web page by default) are removed from what the gateway relays, and the
server is started with `--cors-origins` limited to its own origin (one value: the option does not take a list) when the binary supports it. Requests to `/health`,
`/v1/models` and `OPTIONS` never load the model.

With `apiKey` set in `model-launch.json` (or `gatewayApiKey` in `guardian.json`), everything that changes something or could wake the
model needs `Authorization: Bearer <key>`: the switches under `/guardian/`, every POST/PUT/DELETE, and the monitor's data.
Reading `GET /guardian/status`, `GET /health` and `/v1/models` stays open. The key is handed to llama-server through a private file
(`~/.local/state/ai-gateway/llama-api-key`, mode 600), not on its command line (if the file cannot be written, the command line is
used as before). `shura off/on/unload/warm` send the key from `SHURA_API_KEY` or `model-launch.json`.

Limits against a local program that misbehaves (all in `guardian.json`): `maxRequestBytes` (32 MiB), `maxConnections` (64),
`requestHeaderSeconds` (30: all the headers of a request must arrive within this), `requestBodySeconds` (120), `clientIdleSeconds`
(300, an idle keep-alive connection), `upstreamIdleSeconds` (600 without a byte from a running answer), `allowedHosts`,
`allowedOrigins`. A client that goes away while the model is still thinking has its generation stopped (`watchClientLeave`, on by
default); a client that half-closes its sending side and keeps waiting (an HTTP/1.0 or `Connection: close` request is exempt; an
HTTP/1.1 `nc -N` is not) would be cut off, so set `watchClientLeave` to `false` if you have such a tool.

## The agent runs commands on your machine

ShuraCode executes shell commands and edits files with your user's permissions. Its shipped config
([hikkian/shuracode](https://github.com/hikkian/shuracode)) asks before any shell command that is not on an
allowlist, and denies destructive ones (`rm -rf`, `mkfs`, `dd`, `shutdown`, …). Review that list before relaxing it. Content from web
pages, search results and MCP tools can contain prompt-injection attempts; treat an agent that has
read untrusted content with the same care as untrusted code.

## What the project stores locally

- `~/.local/state/ai-gateway/slots/`: at most one KV-cache file per mode (text/vision). It contains the
  model's internal state for your last long session, which can encode its content. Delete the folder
  if that matters to you.
- `config/guardian.json`, `config/model-launch.json`: machine-local paths. They are gitignored.
- ShuraCode's memory (`~/.local/share/shuracode/memory/`) and its engine's session history
  (`~/.local/share/opencode/`).

No telemetry is sent by any component in this repository. ShuraCode disables its engine's update
checks, sharing and model-catalogue downloads.

## Reporting a vulnerability

Open a private security advisory on this repository (Security → Advisories → Report a vulnerability).
Please do not file public issues for security problems.
