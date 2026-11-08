#!/usr/bin/env python3
"""score.py <evaldir> <variant> <task> <rep> <workdir> <secs> <rc> -> one JSON line"""
import json, os, re, shutil, subprocess, sys

E, V, T, R, W, SECS, RC = sys.argv[1:8]
proj = os.path.join(W, "proj")
events = []
for line in open(os.path.join(W, "events.jsonl"), errors="replace"):
    try:
        events.append(json.loads(line))
    except ValueError:
        pass

texts = [e["part"].get("text", "") for e in events if e.get("type") == "text"]
answer = texts[-1] if texts else ""
if os.path.exists(os.path.join(W, "last.txt")):  # codex exec -o
    answer = open(os.path.join(W, "last.txt"), errors="replace").read()
all_text = "\n".join(texts)
tools = [e["part"].get("tool") for e in events if e.get("type") == "tool_use"]
tool_err = sum(1 for e in events if e.get("type") == "tool_use" and e["part"].get("state", {}).get("status") == "error")
steps = [e["part"].get("tokens", {}) for e in events if e.get("type") == "step_finish"]
out_tok = sum(s.get("output", 0) + s.get("reasoning", 0) for s in steps)
items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]  # codex exec --json
if items:
    tools = [i.get("type") for i in items if i.get("type") in ("command_execution", "file_change", "web_search", "mcp_tool_call")]
    tool_err = sum(1 for i in items if i.get("type") == "command_execution" and i.get("exit_code") not in (0, None))
    out_tok = sum(e["usage"].get("output_tokens", 0) + e["usage"].get("reasoning_output_tokens", 0)
                  for e in events if e.get("type") == "turn.completed" and e.get("usage"))
open(os.path.join(W, "answer.txt"), "w").write(answer)

# protected files: tests shipped with the task must stay unchanged
diff = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=proj, capture_output=True, text=True).stdout.split()
# adding tests is fine; removing or changing existing test lines (or deleting a test file) is not
numstat = subprocess.run(["git", "diff", "--numstat", "HEAD"], cwd=proj, capture_output=True, text=True).stdout.splitlines()
tests_touched = [l.split("\t")[2] for l in numstat
                 if l.split("\t")[2].startswith(("tests/", "test/")) and l.split("\t")[1] not in ("0",)]
changed = diff + subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=proj,
                                capture_output=True, text=True).stdout.split()

# hidden tests (run on a copy so they don't pollute the project)
chk = os.path.join(W, "check")
shutil.rmtree(chk, ignore_errors=True)
shutil.copytree(proj, chk, ignore=shutil.ignore_patterns(".git"))
shutil.copy(os.path.join(E, "tasks", T, "hidden_test.py"), chk)
# visible tests: the ORIGINAL shipped test files must pass against the agent's code
for f in subprocess.run(["git", "ls-files", "tests", "test"], cwd=proj, capture_output=True, text=True).stdout.split():
    orig = subprocess.run(["git", "show", "HEAD:" + f], cwd=proj, capture_output=True).stdout
    os.makedirs(os.path.dirname(os.path.join(chk, f)), exist_ok=True)
    open(os.path.join(chk, f), "wb").write(orig)
env = {**os.environ, "ANSWER_FILE": os.path.join(W, "answer.txt")}
h = subprocess.run([sys.executable, "-m", "unittest", "hidden_test"], cwd=chk, capture_output=True, text=True, env=env, timeout=120)
hidden_ok = h.returncode == 0
# visible tests
if os.path.isdir(os.path.join(chk, "tests")):
    v = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=chk, capture_output=True, text=True, timeout=120)
    visible_ok = v.returncode == 0 and "Ran 0 tests" not in v.stderr
    if not visible_ok:  # the task does not name a framework: pytest-style tests are just as valid (pytest also runs unittest ones)
        v = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=chk, capture_output=True, text=True, timeout=120)
        visible_ok = v.returncode == 0  # rc 5 = no tests collected
elif os.path.isdir(os.path.join(chk, "test")):
    v = subprocess.run(["node", "--test"], cwd=chk, capture_output=True, text=True, timeout=120)
    visible_ok = v.returncode == 0
else:
    visible_ok = None

extra = {}
if T == "t7_question":
    extra["no_edits"] = not changed
if T == "t3_duration":
    extra["wrote_tests"] = os.path.exists(os.path.join(proj, "tests", "test_timeutil.py"))

letters = re.findall(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", re.sub(r"`[^`]*`|```.*?```", "", answer, flags=re.S))
cyr = sum(1 for c in letters if not c.isascii()) / len(letters) if letters else 0.0

ok = hidden_ok and (visible_ok is not False) and not tests_touched and all(extra.values())
print(json.dumps({"variant": V, "task": T, "rep": int(R), "pass": ok, "hidden": hidden_ok, "visible": visible_ok,
                  "tests_touched": tests_touched, **extra, "secs": int(SECS), "rc": int(RC), "n_tools": len(tools),
                  "tool_errors": tool_err, "out_tokens": out_tok, "answer_chars": len(answer), "ru_ratio": round(cyr, 2),
                  "hidden_err": "" if hidden_ok else (h.stdout + h.stderr)[-300:]}, ensure_ascii=False))
