# Coding-harness comparison (Linux, local model)

Four open-source agent harnesses were installed and driven against the same local Tiel-Coder server.
The goal was to pick a daily driver for a fully local, 200k-context coding workstation.

## Practical tests

1. **Bug fix:** a Python file with a planted bug plus its failing pytest file. The harness must find
   the bug, edit the file, run the tests itself and report. Scored by re-running pytest afterwards.
2. **Web search** through an MCP search tool (private SearXNG), with a source citation, for a fact
   newer than the model's training data.
3. **Browser control** through Playwright MCP: open real pages and report their content.

## Results

| | OpenCode 1.18 | Pi 0.87 | Pithagoras | DeepSeek Harness 0.1.5-rc.3 |
|---|---|---|---|---|
| Bug fix | 5/5 (after path fix; 4/6 before) | 3/3 | same engine as Pi | 3/3 |
| Web search via MCP | ✅ correct, cited | via `pi-mcp-adapter` | via `pi-mcp-adapter` + UI | via `dsh-mcp-client` |
| Browser via MCP | ✅ real page content | via adapter | built-in browser add-on | via plugin |
| Command approval | ✅ allow / ask / deny per pattern | ❌ none by design | ❌ (prompt-injection taint guard only) | ✅ |
| Sandbox | ❌ | ❌ | optional per-task container | ✅ workspace-write |
| Interface | terminal | terminal | web UI + Telegram/Slack channels | web UI + TUI |
| Maturity | stable v1, very active | stable | working; channels untested by author | **developer preview**, breaking changes announced |
| Phones home by default | yes → disabled via config/env | no | no | telemetry `FEEDBACK_ONLY` → `DISABLED` |

Only OpenCode's web-search and browser tests were run end to end; the other columns list the
mechanism each harness provides.

**Choice: OpenCode.** It is the most mature, has granular per-command permissions, and all three
practical tests pass against the local model.

## Findings worth knowing

- **OpenCode + quantized model = mistyped absolute paths.** OpenCode's file tools take absolute paths,
  so the model has to reproduce the home directory from the system prompt, and IQ4_XS occasionally
  garbles it (`/home/user` → `/home/usr`). OpenCode then correctly rejects the path as "outside the
  project". Pi and DSH accept relative paths and never hit this. Fixed by one rule in
  [`opencode/AGENTS.md`](../opencode/AGENTS.md) ("always use project-relative paths"): 5/5 afterwards,
  and faster.
- **OpenCode needs to be told it is offline.** At startup it contacts `api.opencode.ai`. When that call
  stalls, `opencode run` hangs forever *before sending anything to the model*. Setting
  `OPENCODE_DISABLE_AUTOUPDATE / _SHARE / _MODELS_FETCH / _DEFAULT_PLUGINS=1` plus `"autoupdate": false`
  and `"share": "disabled"` fixes it for good.
- **Pithagoras is more than a web wrapper around Pi.** It has server-owned runs (close the tab, come
  back later), MCP management through `pi-mcp-adapter`, a persistent-login browser add-on, a
  prompt-injection taint guard and per-session KV disk caching. Its disk cache saves the full slot
  (~780 MB at 187k) after *every* response, which is heavy SSD wear for agentic use. llama-server's
  RAM prompt cache gives the same resume speed with no disk writes, and works with any harness.
- **DeepSeek Harness** has the best-designed safety model (sandbox + approvals). It is still a
  developer preview with open long-context-overflow bugs, and it enables telemetry by default.
