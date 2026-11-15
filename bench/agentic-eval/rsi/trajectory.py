"""RSIL0b: the trajectories an editor reads (RSIP1 report, §4). The compact view is the one Fable measured in reports/rsip1/compact_trajectory.py
(four times shorter than the raw JSON log: median 9 k characters against 36 k bytes); it is copied here so that the package does not depend
on the reports folder. Not shown: hidden-test verdicts. The agent log has none, and the header carries no `resolved` and no `score`.

Selection: runs with visible failure signs first (the project's tests not run, a timeout, a non-zero exit, one command repeated three
times or more), then successful runs for contrast; at most PER_FAMILY runs per family; the languages alternate; a character budget
(TRAJECTORY_CHARS) stops the pool. The pool is written as files, one per run, into the editor's trajectories/ folder.
"""
import collections
import json
from pathlib import Path

OUT_HEAD = 600          # characters of a normal tool output kept (half at the start, half at the end)
ERR_HEAD = 1200         # for a failed call (non-zero exit, error status, a traceback)
TEXT_MAX = 1500         # of one assistant text part
USER_MAX = 2500         # of the task text
TRAJECTORY_CHARS = 120000   # RSIP1 §4: the pool's budget (about 13 median compact trajectories)
PER_FAMILY = 2
REPEAT_LIMIT = 3


def _cut(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars cut]...\n" + text[-half:]


def compact_log(path):
    """The compact text of one OpenCode event log, the number of steps, and the bash commands in order (for the repeat check)."""
    lines, steps, commands = [], 0, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("type")
            part = ev.get("part") or {}
            if kind == "user":
                lines.append("[user] " + _cut(str(ev.get("text") or part.get("text") or ""), USER_MAX))
            elif kind == "tool_use":
                state = part.get("state") or {}
                inp = state.get("input") or {}
                tool = part.get("tool")
                if tool == "bash":
                    head = "$ " + str(inp.get("command", ""))
                    commands.append(str(inp.get("command", "")).strip())
                elif tool in ("read", "write", "edit", "glob", "grep", "list"):
                    head = f"{tool} {inp.get('filePath') or inp.get('path') or inp.get('pattern') or ''}"
                    if tool == "write":
                        head += f" ({len(str(inp.get('content', '')))} chars)"
                else:
                    head = f"{tool} {json.dumps(inp, ensure_ascii=False)[:200]}"
                out = str(state.get("output") or "")
                meta = state.get("metadata") or {}
                exit_code = meta.get("exit")
                failed = exit_code not in (None, 0) or state.get("status") == "error" or "Traceback" in out or "Error" in out[:300]
                tag = f"[tool {tool}" + (f" exit={exit_code}" if exit_code is not None else "") + \
                    (" ERROR" if state.get("status") == "error" else "") + "]"
                lines.append(f"{tag} {head}\n{_cut(out, ERR_HEAD if failed else OUT_HEAD)}")
            elif kind == "text":
                lines.append("[assistant] " + _cut(str(part.get("text") or ""), TEXT_MAX))
            elif kind == "step_finish":
                steps += 1
                tok = part.get("tokens") or {}
                lines.append(f"[step tokens in={tok.get('input')} out={tok.get('output')} reasoning={tok.get('reasoning')} "
                             f"reason={part.get('reason')}]")
    return {"text": "\n".join(lines), "steps": steps, "commands": commands}


def failure_signs(rec, commands):
    """Visible reasons a run failed (no verdict is used): the project's tests were not run, it timed out, it exited non-zero, or one
    command was repeated REPEAT_LIMIT times or more."""
    signs = []
    if not rec.get("tests_run_project"):
        signs.append("project tests not run")
    if rec.get("timed_out"):
        signs.append("timeout")
    if rec.get("agent_exit") not in (None, 0):
        signs.append(f"exit={rec.get('agent_exit')}")
    counts = collections.Counter(c for c in commands if c)
    if counts and max(counts.values()) >= REPEAT_LIMIT:
        signs.append(f"command repeated {max(counts.values())}x")
    return signs


def header(rec, lang, steps, extra=""):
    finished = (not rec.get("timed_out")) and rec.get("agent_exit") in (None, 0)
    return (f"## {rec.get('task')} | {rec.get('family')} | {rec.get('kind')} | {lang} | finished: {'yes' if finished else 'no'} | "
            f"project tests run: {'yes' if rec.get('tests_run_project') else 'no'} | adhoc checks: {rec.get('adhoc_checks', 0)} | "
            f"steps {steps} | {rec.get('seconds')} s" + (f" | {extra}" if extra else ""))


def entry(rec, task, log_path, *, repeats=1, others=()):
    """One pool entry: the header and the compact body of one run, with the failure signs that rank it. RSIL0e E1/E4: `others` are the
    records of the other repeats of the same base task. The header says which repeat this is, what the other one did, and whether the
    repeats disagree; a run whose repeats disagree also ranks among the failures, so both of its repeats reach the pool first."""
    comp = compact_log(log_path)
    rep = rec.get("rep", 1)
    signs = failure_signs(rec, comp["commands"])
    extra = [f"rep {rep} of {repeats}"]
    for o in others:
        extra.append(f"other repeat: {'resolved' if o.get('resolved') else 'not resolved'}, {o.get('seconds')} s")
    if any(bool(o.get("resolved")) != bool(rec.get("resolved")) for o in others):
        extra.append("repeats disagree")
        signs = signs + ["repeats disagree"]
    text = header(rec, task.get("lang", "?"), comp["steps"], " | ".join(extra)) + "\n" + comp["text"] + "\n"
    return {"id": f"{rec['task']}-rep{rep}", "family": rec.get("family"), "lang": task.get("lang", "?"), "fail": signs,
            "text": text, "size": len(text)}


def _interleave(group):
    by_lang = collections.defaultdict(list)
    for e in sorted(group, key=lambda e: e["id"]):
        by_lang[e["lang"]].append(e)
    out = []
    while any(by_lang.values()):
        for lang in sorted(by_lang):
            if by_lang[lang]:
                out.append(by_lang[lang].pop(0))
    return out


def select(entries, budget=TRAJECTORY_CHARS, per_family=PER_FAMILY):
    """Failures first, then successes for contrast; at most `per_family` per family; languages alternate; the budget is not exceeded."""
    fails = _interleave([e for e in entries if e["fail"]])
    passes = _interleave([e for e in entries if not e["fail"]])
    chosen, per, total, cut = [], collections.Counter(), 0, False
    for e in fails + passes:
        if per[e["family"]] >= per_family:
            continue
        if total + e["size"] > budget:
            cut = True
            continue
        chosen.append(e)
        per[e["family"]] += 1
        total += e["size"]
    return {"chosen": chosen, "chars": total, "cut": cut, "left_out": [e["id"] for e in entries if e not in chosen]}


def write_pool(folder, chosen):
    folder = Path(folder)
    if folder.exists():
        for old in folder.glob("*.txt"):
            old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    for n, e in enumerate(chosen, 1):     # the number is the order of selection: failure signs first (RSIL0d P3)
        (folder / f"{n:02d}-{e['id']}.txt").write_text(e["text"], encoding="utf-8")
    return folder
