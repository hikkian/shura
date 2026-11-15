#!/usr/bin/env python3
"""LHSET1 metrics (bench/agentic-eval/lh/metrics_lh.py), counted by script from a run's agent log (opencode events), the grade's JSON parts
and the work tree. No judge. Used by rsi2 for the LH final block and standalone:
    metrics.py AGENT_LOG [--work DIR] [--rule RULE_SCRIPT] [--pre-tests JSON] [--post-tests JSON] -> JSON
1. hallucinated_paths: file tool calls on paths that do not exist (File not found / ENOENT / refused mistyped absolute path), counted
   until the first successful read of the same basename; hallucinated_symbols: imports/requires in written files of modules that are
   neither in the work tree nor standard/vendored (best effort: Python and JS relative imports).
2. start_rule_violation: 0/1 from a rule script run on the final tree (LH-D: the script is part of the task).
3. rework: identical tool calls repeated (same tool + same args) and files edited back and forth (an edit that restores an earlier
   content hash); loops: the longest run of identical consecutive calls; compactions (context summaries).
4. regressions: tests that passed before the run and fail after (needs --pre-tests/--post-tests JSON {name: pass}).
5. unverified_done: the final assistant text claims success (tests pass / done / готово) while no test command ran after the last edit.
"""
import argparse, collections, hashlib, json, os, re, subprocess, sys
from pathlib import Path

DONE_RE = re.compile(r"(all tests pass|tests pass|passed|готово|выполнено|done\b|complete[d]?\b|работает|implemented)", re.I)
TEST_RE = re.compile(r"\b(pytest|unittest|node --test|npm test|jest|vitest|cargo test|go test|make test|\.tmp/\S*(test|probe|check))", re.I)
NOTFOUND_RE = re.compile(r"File not found|ENOENT|No such file|not found: /|specified a rule which prevents", re.I)


def events(log_path):
    with open(log_path, errors="replace") as f:
        lines = f.readlines()
    for line in lines:
        try:
            yield json.loads(line)
        except ValueError:
            continue


def compute(log_path, work=None, rule=None, pre=None, post=None):
    tools = []
    reasoning_compactions = 0
    last_text = ""
    for e in events(log_path):
        t, part = e.get("type"), e.get("part") or {}
        if t == "tool_use":
            st = part.get("state") or {}
            tools.append({"tool": part.get("tool"), "args": st.get("input") or {}, "status": st.get("status"),
                          "out": str(st.get("output") or "") + str(st.get("error") or "")})
        elif t == "text":
            last_text = part.get("text") or last_text
        elif t == "reasoning" and (part.get("text") or "").startswith("Let me analyze the conversation"):
            reasoning_compactions += 1
    # 1. hallucinated paths
    seen_ok = set()
    halluc_paths = []
    for c in tools:
        path = str(c["args"].get("filePath") or c["args"].get("path") or "")
        if c["tool"] in ("read", "edit", "write", "glob", "list") and path:
            base = os.path.basename(path.rstrip("/"))
            if c["status"] == "completed" and not NOTFOUND_RE.search(c["out"][:300]):
                seen_ok.add(base)
            elif NOTFOUND_RE.search(c["out"][:400]) and base not in seen_ok:
                halluc_paths.append(path)
    halluc_symbols = []
    if work and Path(work).is_dir():
        tree = {p.relative_to(work).as_posix() for p in Path(work).rglob("*") if p.is_file()}
        stems = {Path(p).stem for p in tree} | {Path(p).parent.name for p in tree}
        for c in tools:
            if c["tool"] in ("write", "edit"):
                content = str(c["args"].get("content") or c["args"].get("newString") or "")
                for m in re.finditer(r"^\s*from \.(\w+) import|^\s*import (\w+)$|require\(['\"]\./([\w/]+)['\"]\)|from ['\"]\./([\w/]+)['\"]", content, re.M):
                    name = next(g for g in m.groups() if g)
                    if name.split("/")[-1] not in stems and name not in ("json", "os", "sys", "re", "math", "typing", "collections", "itertools", "functools", "pathlib", "unittest", "dataclasses", "string", "random", "datetime", "time", "io", "csv", "hashlib", "struct", "abc", "enum", "copy", "heapq", "bisect", "operator", "textwrap", "argparse", "subprocess", "shutil", "tempfile", "logging", "decimal", "fractions", "statistics", "array", "html", "urllib", "base64", "binascii", "codecs", "unicodedata", "secrets", "contextlib", "inspect", "types", "weakref", "pprint", "glob", "fnmatch"):
                        halluc_symbols.append(name)
    # 2. start rule
    rule_violation = None
    if rule and work:
        r = subprocess.run(["python3", "-I", rule, str(work)], capture_output=True, text=True, timeout=60)
        rule_violation = 0 if r.returncode == 0 else 1
    # 3. rework / loops
    sig = [(c["tool"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False)[:2000]) for c in tools]
    repeats = sum(n - 1 for n in collections.Counter(sig).values() if n > 1)
    longest = cur = 1
    for i in range(1, len(sig)):
        cur = cur + 1 if sig[i] == sig[i - 1] else 1
        longest = max(longest, cur)
    hist = collections.defaultdict(list)
    back_and_forth = 0
    for c in tools:
        if c["tool"] == "write":
            h = hashlib.sha1(str(c["args"].get("content") or "").encode()).hexdigest()
            f = c["args"].get("filePath")
            if h in hist[f][:-1]:
                back_and_forth += 1
            hist[f].append(h)
    # 4. regressions
    regressions = None
    if pre is not None and post is not None:
        regressions = sorted(n for n, ok in pre.items() if ok and not post.get(n, False))
    # 6. recovery cost after a compaction: tool calls between a compaction and the next edit, and re-reads of files read before it
    recovery = []
    compaction_at = []
    idx = 0
    reads_before = set()
    for e in events(log_path):
        t, part = e.get("type"), e.get("part") or {}
        if t == "reasoning" and (part.get("text") or "").startswith("Let me analyze the conversation"):
            compaction_at.append(idx)
        if t == "tool_use":
            idx += 1
    if compaction_at:
        seq = []
        for e in events(log_path):
            if e.get("type") == "tool_use":
                part = e.get("part") or {}
                seq.append((part.get("tool"), str((part.get("state") or {}).get("input", {}).get("filePath") or "")))
        for c in compaction_at:
            before = {f for (tl, f) in seq[:c] if tl == "read" and f}
            calls = 0
            rereads = 0
            for tl, f in seq[c:]:
                calls += 1
                if tl == "read" and f in before:
                    rereads += 1
                if tl in ("write", "edit"):
                    break
            recovery.append({"calls_to_next_edit": calls, "rereads": rereads})
    # 5. unverified done
    last_edit = max((i for i, c in enumerate(tools) if c["tool"] in ("write", "edit")), default=-1)
    last_test = max((i for i, c in enumerate(tools) if c["tool"] == "bash" and TEST_RE.search(str(c["args"].get("command") or ""))), default=-1)
    claims = bool(DONE_RE.search(last_text or ""))
    unverified_done = int(claims and last_edit >= 0 and last_test < last_edit)
    return {"hallucinated_paths": len(halluc_paths), "hallucinated_paths_list": halluc_paths[:10], "hallucinated_symbols": sorted(set(halluc_symbols)),
            "start_rule_violation": rule_violation, "repeated_calls": repeats, "longest_identical_run": longest, "edits_back_and_forth": back_and_forth,
            "compactions": reasoning_compactions, "recovery": recovery, "recovery_calls_total": sum(r["calls_to_next_edit"] for r in recovery),
            "regressions": regressions, "unverified_done": unverified_done,
            "edits": sum(1 for c in tools if c["tool"] in ("write", "edit")), "test_runs": sum(1 for c in tools if c["tool"] == "bash" and TEST_RE.search(str(c["args"].get("command") or ""))),
            "tool_calls": len(tools)}


def from_grade(parts):
    """The grade's JSON parts (check_lh.py) as metrics: regression failures, the start rule, the exact-contract tests."""
    if not isinstance(parts, dict):
        return {}
    out = {}
    reg = parts.get("regression") or {}
    out["regressions"] = list(reg.get("unexpected_failures") or []) if reg else None
    rule = parts.get("rule")
    out["start_rule_violation"] = (0 if rule.get("rc") == 0 else 1) if rule else None
    ex = parts.get("exact_contract") or {}
    out["exact_contract_failed"] = list(ex.get("failed") or [])
    return out


def grade_parts(test_tail):
    """The last JSON line of a check_lh.py output (the harness keeps the tail)."""
    for line in reversed((test_tail or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line).get("parts")
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--work"); ap.add_argument("--rule"); ap.add_argument("--pre-tests"); ap.add_argument("--post-tests")
    a = ap.parse_args()
    pre = json.load(open(a.pre_tests)) if a.pre_tests else None
    post = json.load(open(a.post_tests)) if a.post_tests else None
    print(json.dumps(compute(a.log, a.work, a.rule, pre, post), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
