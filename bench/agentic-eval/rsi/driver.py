"""RSIL0 / RSIL0b: the driver of the RSI cycle. One run keeps its state on disk after every step, so an interrupted run continues where
it stopped. The primitives here (propose, compose, admit) do not write the state; the pilot (rsi/pilot.py) owns the steps.

    python3 -m rsi.driver --run NAME dry                              # RSIL0: each editor gets one short real call (costs about 0.2 USD)
    python3 -m rsi.driver --run NAME status                           # steps done so far
    python3 -m rsi.driver --run NAME pilot --model tiel [--fake] ...  # RSIL0b: the whole cycle (see pilot.py)
    python3 -m rsi.driver --run NAME report                           # the morning report of a pilot
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import accept, budget, editors, sandbox, trajectory

RUNS = Path("/home/alisher/Desktop/AI/reports/rsil1/runs")
USAGE = Path("/home/alisher/Desktop/AI/reports/rsil1/usage.jsonl")
PROMPTS = Path(__file__).resolve().parent / "prompts"
DIFF_RE = re.compile(r"```diff\s*\n(.*?)```", re.S)
MAX_INPUT_CHARS = 60000   # the editor reads everything in its prompt (no tools): the inputs are cut, and the cut is logged
TRAJECTORY_LIMIT = trajectory.TRAJECTORY_CHARS   # RSIP1 §4: the trajectories pool gets at least 120 000 characters


def state_path(name):
    return RUNS / name / "state.json"


def load_state(name):
    p = state_path(name)
    return json.loads(p.read_text()) if p.exists() else {"steps": {}}


def save_state(name, state):
    p = state_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(p)   # a rename, so an interrupted write never leaves half a state file


def extract_diff(text):
    m = DIFF_RE.search(text or "")
    return m.group(1) if m else None


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _opens_next(lines, k):
    """A line that starts a new hunk or a new file. A removed line that starts with `--- ` is not one unless `+++ ` follows it."""
    if lines[k].startswith(("@@ ", "diff --git ")):
        return True
    return lines[k].startswith("--- ") and k + 1 < len(lines) and lines[k + 1].startswith("+++ ")


def normalize_hunks(diff_text: str) -> str:
    """RSIL0e E2 (RSIR2 §1): the editors write the hunk counts by hand and get them wrong. `@@ -0,0 +1,34 @@` above 32 added lines made
    `git apply` refuse the whole 7.9 KB suggestion. Each hunk is recounted from its body: the old count is the context and '-' lines, the
    new count is the context and '+' lines; the start numbers stay. An empty line inside a hunk is an empty context line; empty lines
    before the next header are not part of the hunk. A header that is already right is left as it is."""
    lines = diff_text.split("\n")
    out, k = [], 0
    while k < len(lines):
        m = HUNK_RE.match(lines[k])
        if not m:
            out.append(lines[k])
            k += 1
            continue
        head, k = lines[k], k + 1
        body = []
        while k < len(lines) and not _opens_next(lines, k):
            body.append(lines[k])
            k += 1
        trailing = []
        while body and body[-1] == "":
            trailing.insert(0, body.pop())
        old = new = 0
        for line in body:
            if line.startswith("\\"):          # "\ No newline at end of file" counts for nothing
                continue
            if line.startswith("+"):
                new += 1
            elif line.startswith("-"):
                old += 1
            else:                              # ' ' or an empty line: context
                old += 1
                new += 1
        want_old = int(m.group(2)) if m.group(2) is not None else 1
        want_new = int(m.group(4)) if m.group(4) is not None else 1
        out.append(head if (old, new) == (want_old, want_new) else f"@@ -{m.group(1)},{old} +{m.group(3)},{new} @@{m.group(5)}")
        out.extend(body)
        out.extend(trailing)
    return "\n".join(out)


def apply_diff(base: Path, diff_text: str) -> bool:
    """Apply a unified diff to `base` (a copy, never the live harness). Returns True when it applies cleanly."""
    patch = base.parent / "candidate.diff"
    patch.write_text(normalize_hunks(diff_text), encoding="utf-8")
    # stdin is /dev/null: `patch` asks questions (a reversed patch) and would wait on a terminal; --batch answers them with no
    r = subprocess.run(["git", "apply", "-p1", str(patch)], cwd=base, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        r = subprocess.run(["patch", "-p1", "--forward", "--batch", "-i", str(patch)], cwd=base, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
    patch.unlink(missing_ok=True)
    return r.returncode == 0


def dry_run(name, usage_log=USAGE, keys=("opus", "sonnet", "fable", "sol")):
    """Each editor gets one short real call. The probe asks for a diff that changes one line of a tiny harness; the diff is applied to a
    copy, checked, and reverted (the copy is removed)."""
    state = load_state(name)
    scratch = Path(tempfile.mkdtemp(prefix="rsil0-", dir="/dev/shm"))
    report = {"editors": {}}
    try:
        base = scratch / "harness"
        base.mkdir()
        (base / "AGENTS.md").write_text("Answer in one short sentence.\nSecond line.\n", encoding="utf-8")
        sandbox.assert_clean(base)
        probe = ("You edit a harness file. The file AGENTS.md has two lines: 'Answer in one short sentence.' and 'Second line.'.\n"
                 "Change the second line to 'Second line, edited.' and reply with ONLY a unified diff in one fenced block tagged diff, "
                 "with paths relative to the folder (a/AGENTS.md, b/AGENTS.md).")
        for key in keys:
            role = {"opus": "invariant", "sonnet": "reflector", "fable": "composer", "sol": "invariant"}[key]
            reply = editors.call(key, role, probe, cwd=scratch, usage_log=usage_log, allow_fallback=False)
            diff = extract_diff(reply.text)
            applied = False
            if diff:
                trial = scratch / f"trial-{key}"
                shutil.copytree(base, trial)
                applied = apply_diff(trial, diff) and "edited" in (trial / "AGENTS.md").read_text()
                shutil.rmtree(trial)
            report["editors"][key] = {"model": reply.model, "ok": reply.ok, "diff_found": bool(diff), "diff_applies": applied,
                                      "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out, "cost_usd": round(reply.cost_usd, 4),
                                      "seconds": reply.seconds, "error": reply.error[:200]}
        report["base_unchanged"] = (base / "AGENTS.md").read_text().endswith("Second line.\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    state["steps"]["dry"] = report
    save_state(name, state)
    return report


def read_folder(folder: Path, limit=MAX_INPUT_CHARS):
    """The text of a folder for an editor's prompt: file names and contents, in order, cut at `limit` characters."""
    parts, total, cut = [], 0, False
    for p in sorted(Path(folder).rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or sandbox._forbidden(rel):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        block = f"--- {rel}\n{text}\n"
        if total + len(block) > limit:
            cut = True
            break
        parts.append(block)
        total += len(block)
    return "".join(parts), cut


def _sources(harness_dir, run_dir=None, library_dir=None, trajectories=None):
    """What an editor's folder may hold: the harness, the trajectories (a pool folder of compact texts), the library. Nothing else
    (sandbox.build refuses hidden, solution, verdict, dev and test items)."""
    sources = {"harness": Path(harness_dir)}
    if trajectories is not None and Path(trajectories).exists():
        sources["trajectories"] = Path(trajectories)
    elif run_dir is not None and (Path(run_dir) / "agent").exists():
        sources["trajectories"] = Path(run_dir) / "agent"
    if library_dir is not None and Path(library_dir).exists():
        sources["library"] = Path(library_dir)
    return sources


def propose(name, role, harness_dir, run_dir=None, usage_log=None, editor=None, model="tiel", library_dir=None, cand_dir=None,
            trajectories=None):
    """One editor proposes a diff for `role` ('invariant' or 'specific') on the base harness. The editor works in a sandboxed folder
    (harness, trajectories, library). Writes the reply, the diff and the inputs' cut flags under cand_dir (default RUNS/name/candidates);
    returns the diff or None."""
    usage_log = usage_log or USAGE
    editor = editor or editors.editor_for(role)
    prompt_file = {"invariant": "invariant.md", "cause": "cause.md"}.get(role, "specific.md")   # RSI2D: the cause editor
    prompt_text = (PROMPTS / prompt_file).read_text(encoding="utf-8")
    out = Path(cand_dir) if cand_dir else RUNS / name / "candidates"
    scratch = Path(tempfile.mkdtemp(prefix="rsil-propose-", dir="/dev/shm"))
    try:
        ctx = sandbox.build(scratch / "ctx", _sources(harness_dir, run_dir, library_dir, trajectories))
        harness_text, cut_h = read_folder(ctx / "harness")
        traj, cut_t = read_folder(ctx / "trajectories", TRAJECTORY_LIMIT) if (ctx / "trajectories").exists() else ("", False)
        lib, cut_l = read_folder(ctx / "library") if (ctx / "library").exists() else ("", False)
        prompt = "\n".join([prompt_text, f"# model: {model}", f"# role: {role}", "", "# harness/ (base)", harness_text,
                            "# trajectories/ (train tasks)", traj] + (["# library/", lib] if lib else []))
        reply = editors.call(editor, role, prompt, cwd=ctx, usage_log=usage_log)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    diff = extract_diff(reply.text)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{role}.reply.txt").write_text(reply.text, encoding="utf-8")
    (out / f"{role}.inputs.json").write_text(json.dumps({"cut": {"harness": cut_h, "trajectories": cut_t, "library": cut_l}}))
    if diff:
        (out / f"{role}.diff").write_text(diff, encoding="utf-8")
    elif (out / f"{role}.diff").exists():
        (out / f"{role}.diff").unlink()   # a diff from an earlier round must not reach the composer
    return diff


def compose(name, base_dir, evidence, usage_log=None, model="tiel", library_dir=None, cand_dir=None):
    """The composer (Fable, fallback Opus high) sees the base, the candidate diffs that exist, and the evidence (paired dev results).
    It may keep the base: no diff is returned then (KEEP_BASE is a real answer)."""
    usage_log = usage_log or USAGE
    cand = Path(cand_dir) if cand_dir else RUNS / name / "candidates"
    parts = []
    for r in ("invariant", "specific"):
        if (cand / f"{r}.diff").exists():
            # the author's whole reply: the diff and the note with it (RSIP1 §4: each diff with its note)
            note = (cand / f"{r}.reply.txt").read_text(encoding="utf-8") if (cand / f"{r}.reply.txt").exists() \
                else (cand / f"{r}.diff").read_text(encoding="utf-8")
            parts.append(f"## {r}\n" + note)
    prompt_text = (PROMPTS / "composer.md").read_text(encoding="utf-8")
    scratch = Path(tempfile.mkdtemp(prefix="rsil-compose-", dir="/dev/shm"))
    try:
        ctx = sandbox.build(scratch / "ctx", _sources(base_dir, None, library_dir))
        base_text, _ = read_folder(ctx / "harness")
        lib = read_folder(ctx / "library")[0] if (ctx / "library").exists() else ""
        prompt = "\n".join([prompt_text, f"# model: {model}", "# role: composer", "", "# harness/ (base)", base_text, "# candidates/",
                            *parts, "# evidence/", json.dumps(evidence, ensure_ascii=False, indent=1)]
                           + (["# library/", lib] if lib else []))
        reply = editors.call("fable", "composer", prompt, cwd=ctx, usage_log=usage_log)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    diff = extract_diff(reply.text)
    cand.mkdir(parents=True, exist_ok=True)
    (cand / "composed.reply.txt").write_text(reply.text, encoding="utf-8")
    if diff:
        (cand / "composed.diff").write_text(diff, encoding="utf-8")
    return diff


def admit(name, base_dir, diff_text, cand_results, base_results, allow_stub=False, thresholds=None):
    """Apply a diff to a copy of the base, check its size budget, then decide by our measurements (accept.decide). Nothing is written to
    the live harness. Returns the verdict; the caller records it."""
    trial = RUNS / name / "trial"
    if trial.exists():
        shutil.rmtree(trial)
    shutil.copytree(base_dir, trial)
    size = budget.check(budget.added_lines(diff_text))
    if not size["ok"] or not apply_diff(trial, diff_text):
        return {"accept": False, "reason": "budget" if not size["ok"] else "diff does not apply", "budget": size}
    verdict = accept.decide(cand_results, base_results, thresholds=thresholds, allow_stub=allow_stub)
    verdict["budget"] = size
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi.driver")
    ap.add_argument("--run", required=True)
    ap.add_argument("step", choices=["dry", "status", "pilot", "report", "cycle2", "simulate", "lhcal", "lifelong"])
    ap.add_argument("--model", default="tiel", choices=["tiel", "sage", "gemma"])
    ap.add_argument("--fake", action="store_true", help="no GPU, no paid editor: fake server and stub editors (RSIL0b)")
    ap.add_argument("--train", type=int, default=12, help="train tasks in the cycle (the night's default, RSIR1 D10)")
    ap.add_argument("--agent", default="Claude Haiku 5.5 (2)", help="the name this session holds GPU and cards under on the board")
    ap.add_argument("--dev", type=int, default=0,
                    help="dev tasks for the checkpoints (paired measurements); 0 = every task of --dev-scope (48 for the pilot)")
    ap.add_argument("--dev-scope", default="subset", choices=["subset", "all"],
                    help="subset: the six pre-registered dev ideas of rsi/dev_subset.json (the pilot); all: the whole dev part (RSI1)")
    ap.add_argument("--test", type=int, default=0, help="test tasks for the final test (used once, reported only); 0 = none")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="a checkpoint after this many composed versions (0: only at the end of the train tasks)")
    ap.add_argument("--repeats", type=int, default=2, help="runs per task and arm (at least 2, DATASET.md §5)")
    ap.add_argument("--timebox-min", type=float, default=None, help="stop cleanly after this many minutes")
    ap.add_argument("--editors", default="opus", help="RSI1: the main editors, alternating by train task, e.g. opus,sol")
    ap.add_argument("--max-calls", default="", help="per editor, e.g. opus=40,fable=10,sonnet=40")
    ap.add_argument("--max-tokens", type=int, default=None, help="per editor: input plus output tokens from the usage log")
    ap.add_argument("--stand", default="gateway", choices=["gateway", "server"],
                    help="real stand: the gateway of a pinned worktree (RSI1, gateway_stand.py) or llama-server alone (the pilot's)")
    ap.add_argument("--gateway-worktree", default="/dev/shm/RSI1/gw", help="gateway stand: the pinned git worktree of shura-integration")
    ap.add_argument("--gateway-commit", default=None, help="gateway stand: the commit the worktree must be at (required)")
    ap.add_argument("--effort", default="medium", help="gateway stand: the effort level set on the stand's gateway (EFFORT1)")
    ap.add_argument("--server-argv", default=None, help="server stand: a JSON list with the llama-server command line")
    ap.add_argument("--gpu-task", default="RSIL1", help="real stand: the task name for barid hold gpu")
    ap.add_argument("--general", default="off", choices=["off", "on"],
                    help="RSI1: run the general-v1 tasks too (fast set screens every edit, the whole general dev part gates a checkpoint, general test at the end)")
    ap.add_argument("--dry-editors", default="opus,sonnet,fable,sol", help="dry: which editors get the one short call (a subset spares a limited subscription)")
    ap.add_argument("--crash-after", default=None, help="test hook: raise after this step key, to check resuming")
    ap.add_argument("--skip-task", action="append", default=[],
                    help="a dev task id left out of the checkpoint for both arms (paired): RSIL1-b, VRAM guard after the repeated attempts")
    ap.add_argument("--skip-reason", default="", help="the reason written to the state with --skip-task")
    # RSI2D (rsi/rsi2.py): the cause-driven night and its power simulation
    ap.add_argument("--max-candidates", type=int, default=3, help="cycle2: causes proposed per night (K of the Bonferroni rule)")
    ap.add_argument("--min-room", type=float, default=0.1, help="cycle2: a growth task has a base mean <= 1 - this")
    ap.add_argument("--max-growth", type=int, default=None, help="cycle2: at most this many growth tasks (most room first)")
    ap.add_argument("--prior-runs", default="", help="cycle2: comma list of earlier run names whose base runs feed cause mining and the sets")
    ap.add_argument("--cause-editor", default="opus", help="cycle2: the editor that writes the one edit per cause")
    ap.add_argument("--sim-cycles", type=int, default=300, help="simulate: cycles per cell")
    # RSI2E
    ap.add_argument("--mechanisms", default="", help="cycle2: mechanism plugins as candidates, comma list of " + ",".join(sorted(["requirements-lock", "task-state", "compact-preserve", "done-gate", "action-forcing"])))
    ap.add_argument("--safety-ru", type=int, default=0, help="cycle2: Russian role tasks (tasks_roles, dev part) added to the safety set")
    ap.add_argument("--screen-reps", type=int, default=0, help="cycle2: screen the pool with this many runs per task first and run the full base only on the chosen sets (0 = base on the whole pool)")
    ap.add_argument("--engine-context", type=int, default=None, help="the engine's limit.context (default: the product's 200000; the old pilot used 32768)")
    ap.add_argument("--lh-part", default="dev", choices=["dev", "test"], help="lhcal: which LH part to run")
    ap.add_argument("--lh-n", type=int, default=8, help="lhcal: at most this many LH tasks")
    ap.add_argument("--lh-final", type=int, default=0, help="cycle2 (LHSET1): run this many long-horizon dev tasks (tasks_lh) original vs accepted once at the end, reported only")
    # LL4b (rsi/lifelong.py): the gate of the personal layer: one queued rule of the knowledge store against the current layer
    ap.add_argument("--db", default=str(Path.home() / ".local/share/shura/lifelong.sqlite"), help="lifelong: the knowledge store (rules, control tasks, gate runs)")
    ap.add_argument("--skills-dir", default=str(Path.home() / ".local/share/shura/skills"), help="lifelong: the personal layers (EXPERIENCE.md is written there)")
    ap.add_argument("--candidate", default=None, help="lifelong: the queued rule to measure (default: the oldest queued one)")
    ap.add_argument("--ablate", default=None, help="lifelong: measure the layer WITHOUT this enabled rule instead (§4.7 ablation)")
    args = ap.parse_args(argv)
    if args.step == "lifelong":
        from . import lifelong
        return lifelong.run_cli(args)
    if args.step == "simulate":
        from . import rsi2
        rows = json.loads(Path("/home/alisher/Desktop/AI/reports/rsi2d_rows.json").read_text())
        levels, resid = rsi2.noise_from_rows(rows)
        for n in (13, 24, 36):
            for r in (2, 3, 4):
                for k in (1, 3, 6):
                    out = rsi2.simulate(levels, resid, n, r, k, [0.05, 0.10, 0.15, -0.10], cycles=args.sim_cycles)
                    print(json.dumps({"n": n, "r": r, "k": k, **out}))
        return 0
    if args.step == "dry":
        print(json.dumps(dry_run(args.run, keys=tuple(k for k in args.dry_editors.split(",") if k)), ensure_ascii=False, indent=1))
    elif args.step == "status":
        print(json.dumps(load_state(args.run), ensure_ascii=False, indent=1))
    elif args.step in ("pilot", "cycle2", "lhcal"):
        from . import pilot
        return pilot.run_cli(args)
    else:
        from . import pilot
        print(pilot.morning(args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
