"""RSI1: the general-v1 tasks in the improvement cycle (the rule is rsi/accept_general.py; the sets are described in reports/geval1).

A harness edit for a code model must not hurt the general tasks, so the pilot runs them next to the code tasks:
  * fast  - one task per category of the general dev part (15 tasks; the family with the smallest hash, in it the task with the smallest
            hash): screens every composed edit before it becomes a version, so that a gross regression of the general tasks is caught
            early (2 repeats, both arms: about 2 GPU hours per edit);
  * dev   - the whole general dev part (59 tasks): a checkpoint is accepted only if the code verdict AND this one allow it;
  * test  - the general test part: the final test.
Writing, design, document and post tasks are scored by a rubric judge after the run (general_judge.py, Claude Sonnet through the
subscription CLI): `finish_record` attaches the judge's score to the record, so a record is final the moment the run step ends.
"""
import collections
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_GENERAL = HERE.parent / "tasks_general"
SPLITS = HERE.parent / "splits.json"
FAST_FILE = HERE / "general_fast.json"
SEED = 20261021
JUDGE_PASS = 0.75            # general_summary.attach_judgements: a judged answer is "resolved" from this score


def _tasks(root=None):
    from . import executor
    return executor.all_tasks(root or TASKS_GENERAL)


def _web_ok():
    """Web tasks need the SearXNG MCP in the run config (GEVAL1); without GEVAL_SEARXNG=1 they are left out, as run_experiment does."""
    return os.environ.get("GEVAL_SEARXNG") == "1"


def fast_families(tasks=None, fam=None):
    """{category: family}: per category of the general dev part the family with the smallest sha256(seed:idea); the rule is written
    in general_fast.json and checked by a test."""
    fam = fam if fam is not None else json.loads(SPLITS.read_text(encoding="utf-8"))["families"]
    by_cat = collections.defaultdict(dict)
    for t in tasks if tasks is not None else _tasks():
        if fam.get(t["family"]) == "dev" and (not t.get("needs") or _web_ok()):
            by_cat[t.get("category", "?")][t["family"]] = t.get("idea") or t["family"].split("_", 1)[-1]
    out = {}
    for cat, fams in sorted(by_cat.items()):
        out[cat] = min(fams, key=lambda f: hashlib.sha256(f"{SEED}:{fams[f]}".encode()).hexdigest())
    return out


def select_general(root=None, splits=None):
    """{"fast": [...], "dev": [...], "test": [...]} of task dicts, sorted by id. Tasks that need the web are left out unless GEVAL_SEARXNG=1."""
    fam = json.loads(Path(splits or SPLITS).read_text(encoding="utf-8"))["families"]
    tasks = [t for t in _tasks(root) if not t.get("needs") or _web_ok()]
    fast_fam = fast_families(tasks, fam)
    out = {"fast": [], "dev": [], "test": []}
    for t in tasks:
        split = fam.get(t["family"])
        if split == "dev":
            out["dev"].append(t)
        elif split == "test":
            out["test"].append(t)
    for cat, family in fast_fam.items():     # one task of that family, the one with the smallest hash
        in_family = [t for t in out["dev"] if t["family"] == family]
        out["fast"].append(min(in_family, key=lambda t: hashlib.sha256(f"{SEED}:{t['id']}".encode()).hexdigest()))
    return {k: sorted(v, key=lambda t: t["id"]) for k, v in out.items()}


def fast_gate(candidate, base, noise=None):
    """The screen of one edit on the fast set: the edit is refused when the mean score difference (candidate - base, paired by task and
    repeat) falls to -noise or below. The fast set is small, so no interval rule here: a gross drop is caught, the real decision is the
    checkpoint's (accept_general.general_gate on the whole general dev part)."""
    from . import accept_general
    noise = accept_general.NOISE if noise is None else noise
    gate = accept_general.general_gate(usable(candidate), usable(base), noise)
    if not gate.get("pairs"):
        return dict(gate, ok=False, mode="screen")
    gate["ok"] = bool(gate["score"]["mean"] > -noise)
    gate["mode"] = "screen (mean difference above -noise)"
    return gate


def needs_judge(rec):
    return "judge_pending" in (rec.get("test_tail") or "")


def finish_record(rec, task, answers_dir, judge=None):
    """The judge's score into the record of a judged task (the rule of general_summary.attach_judgements: resolved from 0.75, a failed
    precheck halves the score). A judge that gives no score leaves `judge_failed`: such a run is left out of the pairs (`usable`)."""
    if not needs_judge(rec):
        return rec
    rec = dict(rec)
    tag_prefix = f"{rec['task']}-{rec['arm']}-{rec['rep']}-"
    metas = sorted(Path(answers_dir).glob(f"{tag_prefix}*.json")) if answers_dir else []
    if not metas:
        rec.update(judge_failed="no answer handed off", score=0.0, resolved=False)
        return rec
    meta = metas[-1]
    md = meta.with_suffix(".md")
    if judge is None:
        import general_judge
        judge = general_judge.judge_one
    info = json.loads(meta.read_text(encoding="utf-8"))
    png = meta.with_suffix(".png")
    try:
        res = judge(task["dir"], md.read_text(encoding="utf-8"), image=str(png) if png.is_file() else None) if md.is_file() else {"score": None}
    except Exception as err:     # a judge failure is a record, not a crash of the night
        res = {"score": None, "error": repr(err)[:160]}
    score = res.get("score")
    if score is None:
        rec.update(judge_failed=res.get("error") or "no valid judgement", score=0.0, resolved=False)
        return rec
    if not all(ok for _, ok in info.get("pre", [])):
        score = round(score * 0.5, 4)
    rec.update(score=float(score), resolved=score >= JUDGE_PASS, judged=True, judge_criteria=res.get("criteria"))
    return rec


def usable(records):
    """The runs that count in a paired comparison: not the ones whose judge failed (they would be a zero that is not the model's)."""
    return [r for r in records if not r.get("judge_failed")]
