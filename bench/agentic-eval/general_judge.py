"""GEVAL1: the rubric judge for writing tasks — Claude Sonnet 5.5 through the subscription CLI (`shura-qa claude -p`), never a local model.

Scores every answer in an answers folder (what check.py copied there: <tag>.md + <tag>.json) or explicit files, with the task's fixed
rubric (task.json / hidden/expected.json "rubric"), and writes judged.jsonl: {tag, task_dir, score 0..1, criteria, raw}.
Agreement check: `--agreement N` judges the reference and the wrong answers of N writing tasks twice and reports exact and ±1 agreement.

    python3 -I judge.py --answers DIR --out judged.jsonl
    python3 -I judge.py --agreement 20 --out agreement.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
import shutil
import tempfile
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("GEVAL_TASKS_ROOT") or (Path(__file__).resolve().parent / "tasks_general"))
MODEL = "claude-sonnet-5-5"
CLI_PRE = ["shura-qa", "claude", "-p"]
CLI_POST = ["--model", MODEL, "--output-format", "json", "--no-session-persistence", "--tools", ""]   # the prompt must come before --tools (variadic)
PROMPT = """You are a strict but fair grader. Grade the ANSWER to the TASK with the RUBRIC. For every criterion give an integer 0, 1 or 2
(0 = not met, 1 = partly, 2 = fully). Do not reward length. Do not grade anything outside the rubric. Facts listed under FACTS must
appear in the answer for the first criterion to be fully met. Reply with ONLY a JSON object: {"criteria": [<int>, ...], "comment": "<one sentence>"}.

TASK (as given to the model):
<<<
{prompt}
>>>
{materials}
RUBRIC:
{rubric}
{facts}
ANSWER:
<<<
{answer}
>>>"""


IMAGE_NOTE = ("The file ./shot.png in the current directory is a screenshot of the ANSWER rendered in Chromium. FIRST use the Read tool to look at it, "
              "then grade what you SEE (layout, hierarchy, spacing, polish) together with the HTML below.\n\n")


def call(prompt, tries=3, image=None):
    """One judge call. With `image` (a PNG path) the CLI runs in a scratch folder holding it as shot.png with the Read tool allowed."""
    for i in range(tries):
        cwd, extra, post, tmp = "/dev/shm", [], CLI_POST, None
        if image:
            tmp = tempfile.mkdtemp(prefix="geval-judge-", dir="/dev/shm")
            shutil.copy(image, Path(tmp) / "shot.png")
            cwd, extra = tmp, ["--allowedTools", "Read"]
            post = [x if x != "" else "Read" for x in CLI_POST]      # --tools "" -> --tools Read
        r = subprocess.run(CLI_PRE + [prompt] + post + extra, capture_output=True, text=True, timeout=300, cwd=cwd, stdin=subprocess.DEVNULL)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            obj = json.loads(r.stdout)
            text = obj.get("result", "")
        except ValueError:
            text = r.stdout
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0)), text
            except ValueError:
                pass
        time.sleep(3 * (i + 1))
    return None, text


def judge_one(task_dir, answer_text, image=None):
    task_dir = Path(task_dir)
    tj = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    exp = json.loads((task_dir / "hidden" / "expected.json").read_text(encoding="utf-8"))
    crit = exp["rubric"]["criteria"]
    mats = ""
    for p in sorted((task_dir / "start").rglob("*")):
        if p.is_file():
            mats += f"MATERIAL {p.relative_to(task_dir / 'start')}:\n<<<\n{p.read_text(encoding='utf-8')[:4000]}\n>>>\n"
    facts = ("FACTS: " + "; ".join(exp["facts"]) + "\n") if exp.get("facts") else ""
    prompt = (PROMPT.replace("{prompt}", tj["prompt"]).replace("{materials}", mats).replace("{facts}", facts)
              .replace("{rubric}", "\n".join(f"{i + 1}. {c}" for i, c in enumerate(crit))).replace("{answer}", answer_text[:6000]))
    if image:
        prompt = IMAGE_NOTE + prompt
    obj, raw = call(prompt, image=image)
    if not obj or not isinstance(obj.get("criteria"), list) or len(obj["criteria"]) != len(crit):
        return {"score": None, "criteria": None, "raw": raw[-500:], "error": "no valid judgement"}
    pts = [max(0, min(2, int(x))) for x in obj["criteria"]]
    return {"score": round(sum(pts) / (2 * len(pts)), 4), "criteria": pts, "comment": obj.get("comment", "")[:300]}


def judge_answers(answers_dir, out):
    rows = []
    for meta in sorted(Path(answers_dir).glob("*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        md = meta.with_suffix(".md")
        if not md.is_file():
            continue
        pre_ok = all(ok for _, ok in info.get("pre", []))
        fam, kind = info["task_id"].rsplit("-general-", 1)
        task_dir = ROOT / fam / f"general-{kind}"
        png = meta.with_suffix(".png")
        res = judge_one(task_dir, md.read_text(encoding="utf-8"), image=str(png) if png.is_file() else None)
        res.update({"tag": meta.stem, "task_id": info["task_id"], "pre": info.get("pre"), "pre_ok": pre_ok})
        if res["score"] is not None and not pre_ok:
            res["score"] = round(res["score"] * 0.5, 4)   # a failed precheck (language, length, missing facts) halves the judge's score
        rows.append(res)
        print(res["tag"], res["score"], res.get("comment", res.get("error")), flush=True)
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def checker_module():
    """The shared checker (render_metrics lives there): any task's hidden/check.py is a copy of it."""
    import importlib.util
    src = next(ROOT.glob("*/*/hidden/check.py"), None)
    if src is None:
        raise RuntimeError(f"no tasks under {ROOT}")
    spec = importlib.util.spec_from_file_location("check_general", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_shot(task_dir, html_text):
    """Screenshot of an HTML answer for a render task (the same Chromium measurement as check.py), or None."""
    cg = checker_module()
    exp = json.loads((task_dir / "hidden" / "expected.json").read_text(encoding="utf-8"))
    if exp.get("kind") != "render":
        return None
    tmp = Path(tempfile.mkdtemp(prefix="geval-shot-", dir="/dev/shm"))
    (tmp / "index.html").write_text(html_text, encoding="utf-8")
    png = tmp / "shot.png"
    cg.render_metrics({"file": str(tmp / "index.html"), "viewports": exp["viewports"], "screenshot": exp.get("screenshot"), "png": str(png)})
    return str(png) if png.is_file() else None


def agreement(n, out, seed=20261029, glob="gw_*"):
    tasks = sorted(p.parent for p in ROOT.glob(f"{glob}/*/task.json"))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    tasks = tasks[:n]
    rows = []
    for d in tasks:
        tj = json.loads((d / "task.json").read_text(encoding="utf-8"))
        samples = [("reference", (d / "solution" / tj["answer_file"]).read_text(encoding="utf-8"))]
        w = sorted((d / "mutants").glob("wrong*"))
        samples.append(("wrong1", w[0].read_text(encoding="utf-8")))
        for name, text in samples:
            shot = render_shot(d, text)
            a = judge_one(d, text, image=shot)
            b = judge_one(d, text, image=shot)
            if shot:
                shutil.rmtree(Path(shot).parent, ignore_errors=True)
            rows.append({"task": tj["id"], "sample": name, "pass1": a, "pass2": b})
            print(tj["id"], name, a.get("score"), b.get("score"), flush=True)
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    pairs = [(r["pass1"]["score"], r["pass2"]["score"]) for r in rows if r["pass1"].get("score") is not None and r["pass2"].get("score") is not None]
    exact = sum(1 for a, b in pairs if a == b)
    close = sum(1 for a, b in pairs if abs(a - b) <= 0.125)
    refs = [r["pass1"]["score"] for r in rows if r["sample"] == "reference" and r["pass1"].get("score") is not None]
    wrongs = [r["pass1"]["score"] for r in rows if r["sample"] == "wrong1" and r["pass1"].get("score") is not None]
    sep = sum(1 for d in tasks for r1 in rows if False)
    summary = {"pairs": len(pairs), "exact_agreement": round(exact / len(pairs), 3) if pairs else None,
               "within_one_point_agreement": round(close / len(pairs), 3) if pairs else None,
               "reference_mean": round(sum(refs) / len(refs), 3) if refs else None, "wrong_mean": round(sum(wrongs) / len(wrongs), 3) if wrongs else None,
               "reference_above_wrong": sum(1 for d in tasks for a, b in [(next((r["pass1"]["score"] for r in rows if r["task"] == json.loads((d / 'task.json').read_text())["id"] and r["sample"] == "reference"), None),
                                                                           next((r["pass1"]["score"] for r in rows if r["task"] == json.loads((d / 'task.json').read_text())["id"] and r["sample"] == "wrong1"), None))] if a is not None and b is not None and a > b)}
    print(json.dumps(summary, ensure_ascii=False))
    Path(out).with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers")
    ap.add_argument("--agreement", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="gw_*", help="families for --agreement (e.g. 'g[du]_*' for the render tasks)")
    a = ap.parse_args(argv)
    if a.agreement:
        agreement(a.agreement, a.out, glob=a.glob)
    elif a.answers:
        judge_answers(a.answers, a.out)
    else:
        sys.exit("give --answers DIR or --agreement N")


if __name__ == "__main__":
    main()
