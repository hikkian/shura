"""RSIL0: one interface to the editors of the RSI cycle (Claude Opus, Sonnet, Fable through the local `claude -p`, and GPT-6.1 Sol
through `codex exec`). Every call writes one line to a usage log: model, role, tokens, cost, seconds, exit code.

The Claude binary is the one of the local Claude Code install (the same program that runs the session); it is called in print mode
with no tools. Each Claude call carries the CLI's own cached context (about 20–26k tokens, about 0.10–0.55 USD per call depending on
the model): this is measured and written to the usage log, not hidden.

    call(editor="opus", role="invariant", prompt="...", usage_log=Path(...))  ->  Reply(text, ok, tokens_in, tokens_out, cost_usd, ...)

When a call fails because of a subscription limit, the call goes to the editor's fallback (see FALLBACK) and the switch is logged.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

def find_claude(env=None, home=None, which=shutil.which):
    """The Claude Code binary: RSI_CLAUDE_BIN, else `claude` on PATH, else the newest VS Code extension build. The path used to be one pinned
    extension version (2.1.304); the extension updated itself and the first editor call of the night would have died on a missing file
    (found by the RSI1 preflight, 2026-10-31)."""
    env = os.environ if env is None else env
    if env.get("RSI_CLAUDE_BIN"):
        return env["RSI_CLAUDE_BIN"]
    found = which("claude")
    if found:
        return found
    root = Path(home or Path.home()) / ".vscode" / "extensions"

    def version(path):
        m = re.search(r"claude-code-(\d+(?:\.\d+)*)-", path.parent.parent.parent.name)
        return tuple(int(x) for x in m.group(1).split(".")) if m else ()
    builds = sorted(root.glob("anthropic.claude-code-*/resources/native-binary/claude"), key=version)
    return str(builds[-1]) if builds else "claude"


CLAUDE_BIN = find_claude()
CODEX_BIN = os.environ.get("RSI_CODEX_BIN", shutil.which("codex") or "codex")

# editor key -> how to call it
EDITORS = {
    "opus": {"kind": "claude", "model": "claude-opus-5-5", "effort": "medium"},
    "opus-high": {"kind": "claude", "model": "claude-opus-5-5", "effort": "high"},
    "sonnet": {"kind": "claude", "model": "claude-sonnet-5-5", "effort": "medium"},
    "fable": {"kind": "claude", "model": "claude-fable-5-1", "effort": "high"},
    "sol": {"kind": "codex", "model": "gpt-6.1-sol", "effort": "medium"},
}
# the roles of the cycle (the card): editors alternate by task and by role, Opus is the main editor, Sol only when Claude hits its limit
ROLE_EDITOR = {
    "invariant": "opus",
    "specific": "opus",
    "composer": "fable",
    "reflector": "sonnet",
    "selector": "sonnet",
}
FALLBACK = {"opus": "sol", "opus-high": "sol", "fable": "opus-high", "sonnet": "sol", "sol": "opus"}
LIMIT_RE = re.compile(r"(usage limit|rate limit|limit reached|quota|too many requests|credit balance|exceeded.*limit)", re.I)


@dataclass
class Reply:
    editor: str
    model: str
    role: str
    ok: bool
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    exit_code: int = 0
    error: str = ""
    switched_from: str = ""
    extra: dict = field(default_factory=dict)


# The default system prompt of Claude Code describes tools; with `--tools ""` Sonnet and Fable still answered with tool-call text
# (<invoke name="Read">...) instead of the diff (RSI1 preflight, 2026-10-31): an empty edit at night. The editor gets its own short prompt.
EDITOR_SYSTEM = ("You are a text-only editor inside an automated pipeline. You have no tools: you cannot read, write or run anything. "
                 "Everything you need is in the message. Answer with plain text in the exact format the message asks for "
                 "(for example one fenced diff block). Never write tool calls or XML tags such as <invoke> or <parameter>.")
TOOL_MARKUP = re.compile(r"<invoke\b|<parameter\b|antml:")


def _claude_cmd(spec):
    """The prompt comes on stdin: an argument is limited to 131 072 bytes (RSIR1 D1). The isolation flags leave out the user's and the
    project's settings, plugins and skills, and move the per-machine sections out of the system prompt. Measured with a short prompt
    (RSIL0d): input tokens 25 905 -> 2 907 on Opus, 2 858 on Sonnet, 9 537 on Fable; the answers did not change."""
    return [CLAUDE_BIN, "-p", "--model", spec["model"], "--effort", spec["effort"], "--tools", "", "--system-prompt", EDITOR_SYSTEM,
            "--no-session-persistence", "--output-format", "json", "--setting-sources", "project,local",
            "--disable-slash-commands", "--exclude-dynamic-system-prompt-sections"]


def _codex_cmd(spec, out_file):
    """The prompt comes on stdin ('-'); --ignore-user-config and --ignore-rules leave out the user's codex configuration and rules."""
    return [CODEX_BIN, "exec", "-m", spec["model"], "-c", f"model_reasoning_effort={spec['effort']}", "--ephemeral",
            "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "-o", str(out_file), "-"]


def _run(cmd, cwd, timeout, prompt):
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=cwd, input=prompt, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - t0
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\ntimeout", time.monotonic() - t0


def _one(editor, role, prompt, cwd, timeout):
    spec = EDITORS[editor]
    if spec["kind"] == "claude":
        code, out, err, secs = _run(_claude_cmd(spec), cwd, timeout, prompt)
        reply = Reply(editor, spec["model"], role, ok=False, exit_code=code, seconds=round(secs, 2))
        try:
            data = json.loads(out)
        except ValueError:
            reply.error = (err or out)[-400:]
            return reply
        usage = data.get("usage") or {}
        reply.text = data.get("result") or ""
        reply.tokens_in = int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0) + \
            int(usage.get("cache_read_input_tokens") or 0)
        reply.tokens_out = int(usage.get("output_tokens") or 0)
        reply.cost_usd = float(data.get("total_cost_usd") or 0.0)
        reply.ok = code == 0 and not data.get("is_error")
        if not reply.ok:
            reply.error = str(data.get("result") or data.get("api_error_status") or err)[-400:]
        return reply
    # codex: the last message goes to a file (-o), the event stream is in stdout/stderr
    out_dir = Path(tempfile.mkdtemp(prefix="codex-", dir=cwd))
    out_file = out_dir / "last.txt"
    code, out, err, secs = _run(_codex_cmd(spec, out_file), cwd, timeout, prompt)
    reply = Reply(editor, spec["model"], role, ok=code == 0, exit_code=code, seconds=round(secs, 2))
    if out_file.exists():
        reply.text = out_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"tokens used\s*\n\s*([\d,]+)", err + out)
    if m:
        reply.tokens_out = int(m.group(1).replace(",", ""))   # codex reports one total for the whole turn
        reply.extra["tokens_total_codex"] = reply.tokens_out
    if not reply.ok:
        reply.error = (err or out)[-400:]
    shutil.rmtree(out_dir, ignore_errors=True)
    return reply


def log(usage_log, reply):
    if usage_log is None:
        return
    Path(usage_log).parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: v for k, v in asdict(reply).items() if k != "text"}}
    with open(usage_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bad_attempt(reply):
    """A call that must be repeated: it timed out (a hung process), or the editor answered with tool-call markup instead of text."""
    if reply.exit_code == 124 or (not reply.ok and "timeout" in (reply.error or "")):
        return "timeout"
    if reply.ok and TOOL_MARKUP.search(reply.text or ""):
        return "tool markup"
    return ""


def call(editor, role, prompt, cwd, usage_log=None, timeout=900, allow_fallback=True, retries=1):
    """One call. A timed-out call or a reply made of tool-call markup is repeated (`retries` times, the retry noted in `extra`, every attempt
    in the usage log); on a limit error the editor's fallback is used once; the switch is recorded in the reply and the log."""
    reply = _one(editor, role, prompt, cwd, timeout)
    log(usage_log, reply)
    for attempt in range(retries):
        why = _bad_attempt(reply)
        if not why:
            break
        again = prompt + ("\n\nReply with plain text only, in the format asked. You have no tools." if why == "tool markup" else "")
        reply = _one(editor, role, again, cwd, timeout)
        reply.extra["retry"] = why
        log(usage_log, reply)
    if not reply.ok and allow_fallback and LIMIT_RE.search(reply.error or "") and editor in FALLBACK:
        nxt = FALLBACK[editor]
        second = _one(nxt, role, prompt, cwd, timeout)
        second.switched_from = editor
        log(usage_log, second)
        return second
    return reply


def editor_for(role):
    return ROLE_EDITOR[role]
