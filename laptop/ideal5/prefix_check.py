#!/usr/bin/env python3
"""Prefix-cache check for a chat template (llama-server KV reuse).

Usage:
    prefix_check.py <template.jinja> [--engine]

For every dialog in dialogs.py and every assistant message at index i (turn k):
    prompt(k)   = render(messages[:i],  add_generation_prompt=True)
    prompt(k+1) = render(messages[:j],  add_generation_prompt=True)
where messages[i] is the assistant message, and messages[i+1:j] is the next
user message, or the run of consecutive tool messages that answers it.
A turn PASSES when prompt(k) is an exact prefix of prompt(k+1), i.e. the server
can reuse the KV cache for everything prompt(k) already processed.

Default: render with jinja2 (same environment setup as v4check.py).
--engine: start llama-server on 127.0.0.1:8093 (CPU only, stories15M), render
through POST /apply-template, and stop the server at the end. The jinja2
render of the same message list is also computed and compared, and the
differences are listed (diagnostic only; they do not change the breaks count).

Exit codes: 0 = no breaks, 1 = breaks > 0, 2 = render or server error.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
from dialogs import DIALOGS  # noqa: E402

ENGINE_PORT = 8093
BUILD_BIN = os.path.expanduser("~/.local/share/shura/shura.cpp/build-cuda/bin")
MODEL = os.path.expanduser("~/.local/share/shura/models/stories15M-q4_0.gguf")
MIN_FREE_KB = int(os.environ.get("PREFIX_CHECK_MIN_FREE_KB", 4 * 1024 * 1024))  # keep >= 4 GiB MemAvailable; the 15M test server needs ~100 MB, override only while a big model is resident
CONTEXT = 60                  # chars shown before and after the first differing char
MAX_CROSS_LINES = 40          # cap on printed engine-vs-jinja2 differences


class RenderError(Exception):
    pass


def _raise_exception(message):
    raise RenderError(message)


def _tojson(value, indent=None):
    # HF-transformers style: no HTML escaping, non-ASCII kept as is.
    return json.dumps(value, ensure_ascii=False, indent=indent)


class JinjaRenderer:
    name = "jinja2"

    def __init__(self, template_text):
        import jinja2
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = _raise_exception
        env.filters["tojson"] = _tojson
        self.template = env.from_string(template_text)

    def __call__(self, messages, tools, kwargs):
        try:
            return self.template.render(messages=messages, tools=tools,
                                        add_generation_prompt=True, **kwargs)
        except Exception as exc:  # jinja2 raises UndefinedError, TemplateError, RenderError
            raise RenderError(f"jinja2: {exc}") from exc


class EngineRenderer:
    name = "engine"

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}/apply-template"

    def __call__(self, messages, tools, kwargs):
        body = {"messages": messages, "chat_template_kwargs": kwargs}
        if tools:
            body["tools"] = tools
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)["prompt"]
        except urllib.error.HTTPError as exc:
            raise RenderError(f"engine HTTP {exc.code}: {exc.read()[:300]!r}") from exc
        except Exception as exc:
            raise RenderError(f"engine: {exc}") from exc


def turn_list(messages):
    """Return (assistant_index, next_end) pairs; prompt(k)=messages[:i], prompt(k+1)=messages[:next_end].

    next_end is None when the assistant message is the last one (nothing to compare)."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        if i == 0 or messages[i - 1]["role"] not in ("user", "tool"):
            raise ValueError(f"assistant at index {i} is not preceded by a user/tool message")
        if i + 1 >= len(messages):
            out.append((i, None))
            continue
        nxt = messages[i + 1]["role"]
        if nxt == "user":
            end = i + 2
        elif nxt == "tool":
            end = i + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
        else:
            raise ValueError(f"assistant at index {i} followed by role {nxt!r}")
        out.append((i, end))
    return out


def first_diff(a, b):
    """Index of the first differing character, or min(len) when one string is a prefix of the other."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def context_of(s, idx):
    return repr(s[max(0, idx - CONTEXT):idx]), repr(s[idx:idx + CONTEXT])


def run_dialogs(render, cross=None):
    breaks = 0
    errors = 0
    checked = 0
    cross_diffs = []
    cross_total = 0
    for number, dialog in enumerate(DIALOGS, 1):
        tools = dialog.get("tools")
        kwargs = dialog.get("chat_template_kwargs") or {}
        msgs = dialog["messages"]
        turns = turn_list(msgs)
        print(f"dialog {number} [{dialog['name']}] kwargs={kwargs} tools={'yes' if tools else 'no'} "
              f"assistant_turns={len(turns)}")
        passed = failed = 0
        for k, (i, end) in enumerate(turns, 1):
            if end is None:
                print(f"  turn {k} asst=#{i}: SKIP (no following message)")
                continue
            checked += 1
            try:
                a = render(msgs[:i], tools, kwargs)
                b = render(msgs[:end], tools, kwargs)
            except RenderError as exc:
                errors += 1
                print(f"  turn {k} asst=#{i}: ERROR {exc}")
                continue
            if cross is not None:
                for label, sub, eng in (("k", msgs[:i], a), ("k+1", msgs[:end], b)):
                    cross_total += 1
                    try:
                        jin = cross(sub, tools, kwargs)
                    except RenderError as exc:
                        jin = f"<jinja2 error: {exc}>"
                    if jin != eng:
                        idx = first_diff(eng, jin)
                        cross_diffs.append((number, k, label, idx, eng, jin))
            prefix_ok = b.startswith(a)
            head = (f"  turn {k} asst=#{i} k=msgs[:{i}] k+1=msgs[:{end}] "
                    f"len(k)={len(a)} len(k+1)={len(b)}")
            if prefix_ok:
                passed += 1
                print(f"{head}: PASS")
            else:
                breaks += 1
                failed += 1
                idx = first_diff(a, b)
                byte_idx = len(a[:idx].encode("utf-8"))
                before_a, after_a = context_of(a, idx)
                before_b, after_b = context_of(b, idx)
                print(f"{head}: FAIL first differing byte {byte_idx} (char {idx})")
                print(f"      k    before={before_a}")
                print(f"      k    after ={after_a}")
                print(f"      k+1  before={before_b}")
                print(f"      k+1  after ={after_b}")
        print(f"  dialog {number} summary: {passed} pass, {failed} fail")
    return breaks, errors, checked, cross_diffs, cross_total


def print_cross(cross_diffs, cross_total):
    print(f"engine vs jinja2 renders differing: {len(cross_diffs)} of {cross_total}")
    for number, k, label, idx, eng, jin in cross_diffs[:MAX_CROSS_LINES]:
        before_e, after_e = repr(eng[max(0, idx - 50):idx]), repr(eng[idx:idx + 50])
        before_j, after_j = repr(jin[max(0, idx - 50):idx]), repr(jin[idx:idx + 50])
        print(f"  dialog {number} turn {k} {label}: first diff at char {idx}")
        print(f"      engine  before={before_e} after={after_e}")
        print(f"      jinja2  before={before_j} after={after_j}")
    if len(cross_diffs) > MAX_CROSS_LINES:
        print(f"  ... {len(cross_diffs) - MAX_CROSS_LINES} more not shown")


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def mem_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def start_engine(template_path):
    if port_busy(ENGINE_PORT):
        raise RuntimeError(f"port {ENGINE_PORT} is already in use; not touching it")
    free = mem_available_kb()
    if free < MIN_FREE_KB:
        raise RuntimeError(f"refusing to start llama-server: MemAvailable is {free / 1048576:.2f} GiB, "
                           f"policy keeps >= 4 GiB free")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LD_LIBRARY_PATH"] = f"{BUILD_BIN}:/usr/local/cuda/lib64"
    cmd = [os.path.join(BUILD_BIN, "llama-server"), "--port", str(ENGINE_PORT), "-m", MODEL,
           "-ngl", "0", "-c", "4096", "-t", "1", "--jinja", "--chat-template-file", template_path]
    log = tempfile.TemporaryFile(dir=HERE)  # server log, deleted on close
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 30
    while True:
        if proc.poll() is not None:
            stop_engine(proc, log)
            raise RuntimeError("llama-server exited during startup:\n" + server_log_tail(log))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    return proc, log
        except Exception:
            pass
        if time.time() > deadline:
            stop_engine(proc, log)
            raise RuntimeError("llama-server did not report /health ok within 30 s:\n" + server_log_tail(log))
        time.sleep(0.5)


def server_log_tail(log, lines=25):
    log.seek(0)
    text = log.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


def stop_engine(proc, log):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    if log is not None:
        log.close()


def main(argv):
    flags = [a for a in argv[1:] if a.startswith("--")]
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 1 or any(f != "--engine" for f in flags):
        print("usage: prefix_check.py <template.jinja> [--engine]", file=sys.stderr)
        return 2
    engine = "--engine" in flags
    template_path = os.path.abspath(os.path.expanduser(args[0]))
    with open(template_path, encoding="utf-8") as f:
        template_text = f.read()
    jinja = JinjaRenderer(template_text)
    mode = "engine" if engine else "jinja2"
    print(f"== prefix_check template={template_path} mode={mode} ==")
    proc = log = None
    try:
        if engine:
            proc, log = start_engine(template_path)
            print(f"llama-server 127.0.0.1:{ENGINE_PORT} ready (CPU, model={os.path.basename(MODEL)}, "
                  f"chat-template-file={template_path})")
            render = EngineRenderer(ENGINE_PORT)
            cross = jinja
        else:
            render = jinja
            cross = None
        breaks, errors, checked, cross_diffs, cross_total = run_dialogs(render, cross)
        if engine:
            print_cross(cross_diffs, cross_total)
    except Exception as exc:  # server start failure, bad fixture, etc.
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("breaks: incomplete (run aborted)")
        return 2
    finally:
        stop_engine(proc, log)
    print(f"turns checked: {checked}  render errors: {errors}")
    print(f"breaks: {breaks}")
    if breaks > 0:
        return 1
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
