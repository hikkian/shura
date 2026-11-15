"""RSIL0b / RSIL0c / RSIL0d: the RSI cycle for one executor model, end to end and resumable (reports/rsi0/LOOP.md, the RSIL1 card, RSIP1 §4,
the RSIR1 review).

    python3 -m rsi.driver --run NAME pilot --model tiel --fake --train 3 --dev 4 --repeats 2   # no GPU, no paid editor
    python3 -m rsi.driver --run NAME pilot --model tiel [--server-argv FILE]                   # real: GPU and paid editors
    python3 -m rsi.driver --run NAME report                                                    # MORNING.md

Order (every run is a step of its own; a rerun continues from the last saved step):
  baseline-dev: the accepted harness (the original at the start) on the dev tasks, before any edit (2 runs per dev task).
  per train task: the BASE runs of this task under the working harness come first (they are the trajectories); the pool is built from
  the base runs of the last POOL_TASKS train tasks, each made under the harness of its own time (the pool makes no runs of its own);
  the invariant and specific editors propose from the pool; each candidate is measured on this task against the base (repeats);
  the composer decides from those numbers (KEEP_BASE allowed); the composed diff becomes the working harness (the next version).
  checkpoints: after K versions (and at the end) the working harness is measured on the dev tasks against the accepted one (DATASET.md
  §5, cached by fingerprint); accepted: it becomes the accepted harness; refused: the working harness goes back.
  final-test: once, reported only, the original against the accepted harness on the test tasks.

Budgets: the size of a harness after an edit (always-on text: 2 500 tokens, 7 000 characters, against the original), the size of a skill
body or script (4 000 characters), the surface (harness.md, skills, scripts only), per editor calls and tokens, the timebox between runs.
"""
import collections
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import accept, accept_general, budget, driver, editors, executor, general, sandbox, trajectory

HERE = Path(__file__).resolve().parent
TASKS_V3 = HERE.parent / "tasks_v3"
SPLITS = HERE.parent / "splits.json"
DEV_SUBSET = HERE / "dev_subset.json"   # the pre-registered six dev ideas of the pilot (RSIL0c); the full dev part is for RSI1
HARNESS_CFG = Path("/home/alisher/Projects/shuracode-integration/config/harness")
REAL_LIBRARY = Path("/home/alisher/Desktop/AI/reports/rsil1/library")
STUB_EDITOR = HERE / "stub_editor.py"
BARID = "/home/alisher/Projects/barid/skills/barid/scripts/barid.py"
BOARD = "/home/alisher/Desktop/AI/.relayboard/board.json"
AGENT = "Claude Haiku 5.5 (2)"
SEED = 20261021
POOL_TASKS = 13          # train tasks whose base runs a pool may draw on (RSIP1 §4: about 13 median compact trajectories)
SKILL_CHARS = 4000       # the body of one skill or one script, added by one edit
SURFACE = re.compile(r"^(harness\.md|skills/[A-Za-z0-9_.-]+/SKILL\.md|skills/[A-Za-z0-9_.-]+/scripts/[A-Za-z0-9_.-]+)$")
# RSI2E: a mechanism candidate adds <model>/plugins.json (the cycle writes it, never an editor); the names it may hold are checked by executor.plugin_candidates
SURFACE_MECHANISM = re.compile(r"^plugins\.json$")
KINDS = ["implement", "bugfix", "refactor", "tests-only"]
FABLE_KINDS = ("implement", "refactor", "tests-only")   # RSIL0e E3: the kinds where the base is weak (RSIR2 §6); bugfix is 83 % on dev


class Stop(RuntimeError):
    """The cycle stops here, cleanly: a budget or the timebox is used up. Finished steps stay in the state; a rerun continues."""


def editor_for(role, task, base, index=0, editors=("opus",)):
    """RSIL0e E3 (RSIR2 §6): the invariant editor is the main editor. The specific editor is Fable where a skill has something to hold:
    the kinds where the base is weak (implement, refactor, tests-only), a task whose base repeats disagree, and a task where every base
    run failed. Bugfix stays with the main editor. RSI1: the main editor alternates by train task over `editors` (Opus, Sol)."""
    main = editors[index % len(editors)]
    if role != "specific":
        return main
    disagree = len({bool(r.get("resolved")) for r in base}) > 1
    return "fable" if (task.get("kind") in FABLE_KINDS or disagree or not any(r.get("resolved") for r in base)) else main


def usage_rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def family_means(cand, base):
    """Per family: the mean score difference of the candidate against the base, over the paired runs of that family."""
    b = {(r["task"], r.get("rep", 1)): r for r in base}
    per = collections.defaultdict(list)
    for c in cand:
        r = b.get((c["task"], c.get("rep", 1)))
        if r is not None:
            per[c["family"]].append(c["score"] - r["score"])
    return {f: round(sum(v) / len(v), 4) for f, v in sorted(per.items())}


def _summary(rec):
    return {"resolved": bool(rec.get("resolved")), "score": rec.get("score"), "seconds": rec.get("seconds")}


def paired_runs(cand, base):
    """RSIR1 D8: the runs side by side, by repeat: what the base did and what the candidate did."""
    b = {r.get("rep", 1): r for r in base}
    return [{"rep": c.get("rep", 1), "base": _summary(b[c.get("rep", 1)]), "candidate": _summary(c)}
            for c in cand if c.get("rep", 1) in b]


def growth(new_text, orig_text):
    """RSIR1 D6: how much an edit adds to the always-on text, against the original: estimated tokens and characters."""
    tokens = budget.estimate_tokens(new_text) - budget.estimate_tokens(orig_text)
    chars = len(new_text) - len(orig_text)
    return {"tokens": tokens, "chars": chars, "ok": tokens <= budget.BUDGET_TOKENS and chars <= budget.MAX_CHARS}


def _diff_name(line):
    name = line[4:].split("\t")[0].strip()
    if name == "/dev/null":
        return None
    return name[2:] if name[:2] in ("a/", "b/") else name


def diff_files(diff_text):
    """{file: the lines it adds}, and the set of every file the diff names (both sides)."""
    files, names, current = {}, set(), None
    for line in (diff_text or "").splitlines():
        if line.startswith("--- "):
            name = _diff_name(line)
            if name:
                names.add(name)
            current = None
        elif line.startswith("+++ "):
            name = _diff_name(line)
            if name:
                names.add(name)
                files.setdefault(name, [])
            current = name
        elif current and line.startswith("+") and not line.startswith("+++"):
            files[current].append(line[1:])
    return files, names


def surface_ok(diff_text):
    """The surface of a model: harness.md, skills/<name>/SKILL.md, skills/<name>/scripts/<file>. AGENTS.md (shared by every model) and
    subagents (shared by the engine) are not the model's: a diff that names one of them is refused (RSIL0d)."""
    _, names = diff_files(diff_text)
    return bool(names) and all(SURFACE.match(n) or SURFACE_MECHANISM.match(n) for n in names)


def surface_size(diff_text):
    """A skill body or a script added by one edit: SKILL_CHARS at most each (the always-on text is checked by growth)."""
    files, _ = diff_files(diff_text)
    over = {n: len("\n".join(v)) for n, v in files.items() if n != "harness.md" and len("\n".join(v)) > SKILL_CHARS}
    return {"over_limit": over, "ok": not over}


def copy_tree(src, dst):
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def select_tasks(n_train, n_dev, n_test=0, dev_scope="subset", root=TASKS_V3, splits=SPLITS):
    """Train, dev and test tasks by the family split of bench/agentic-eval/splits.json; a fixed seed picks them (a rerun picks the same).
    Per round the least-used family goes first; within a family the kinds rotate (implement, bugfix, refactor, tests-only) so that one
    kind does not take the whole set (RSIR1 D4). dev_scope: "subset" = the pre-registered six dev ideas (the pilot), "all" = the whole dev
    part (RSI1). n_dev and n_test: how many; 0 = every task of the scope (the test part is only used by the final test)."""
    fam = json.loads(Path(splits).read_text())["families"]
    subset = set(json.loads(DEV_SUBSET.read_text(encoding="utf-8"))["ideas"]) if dev_scope == "subset" else None
    by = {"train": collections.defaultdict(list), "dev": collections.defaultdict(list), "test": collections.defaultdict(list)}
    for t in executor.all_tasks(root):
        split = fam.get(t["family"])
        if split not in by:
            continue
        if split == "dev" and subset is not None and t["family"].split("_", 1)[-1] not in subset:
            continue
        by[split][t["family"]].append(t)
    rng = random.Random(SEED)

    def pick(groups, n):
        if not n:
            return sorted((t for g in groups.values() for t in g), key=lambda t: t["id"])
        pools = {f: sorted(ts, key=lambda t: t["id"]) for f, ts in groups.items()}
        for ts in pools.values():
            rng.shuffle(ts)
        order = {f: rng.random() for f in pools}
        kinds = list(KINDS)
        rng.shuffle(kinds)
        used, picked = collections.Counter(), []
        for k in range(n):
            live = [f for f in pools if pools[f]]
            if not live:
                break
            want = kinds[k % len(kinds)]
            f = min(live, key=lambda f: (used[f], order[f]))
            ts = pools[f]
            picked.append(ts.pop(next((i for i, t in enumerate(ts) if t["kind"] == want), 0)))
            used[f] += 1
        return sorted(picked, key=lambda t: t["id"])

    # n_test 0 means no final test at all (RSIL0d): the test part is used only when asked for a number of tasks
    return {"train": pick(by["train"], n_train), "dev": pick(by["dev"], n_dev),
            "test": pick(by["test"], n_test) if n_test else []}


class Pilot:
    def __init__(self, name, model, tasks, runner, *, thresholds, allow_stub=False, repeats=2, limits=None, timebox_min=None,
                 library=None, fake=False, crash_after=None, checkpoint_every=0, dev_scope="subset", editors=("opus",),
                 general=None, judge=None):
        self.name, self.model, self.tasks, self.runner = name, model, tasks, runner
        self.thresholds, self.allow_stub, self.dev_scope = thresholds, allow_stub, dev_scope
        self.repeats, self.limits, self.timebox_min, self.fake = repeats, limits or {}, timebox_min, fake
        self.crash_after, self.checkpoint_every = crash_after, checkpoint_every
        self.editors = tuple(editors)
        # RSI1: the general-v1 tasks next to the code tasks (rsi/general.py): {"fast": [...], "dev": [...], "test": [...]} or None (code only)
        self.general, self.judge = general, judge
        self.dir = driver.RUNS / name
        self.usage = self.dir / "usage.jsonl"
        self.library = Path(library) if library else (self.dir / "library" if fake else REAL_LIBRARY / model)
        # working: the harness being edited (W); accepted: the last checkpoint-accepted one (A); original: the starting one (O)
        self.h_root, self.acc_root, self.orig_root = self.dir / "harness", self.dir / "accepted", self.dir / "original"
        self.h_dir, self.acc_dir, self.orig_dir = self.h_root / model, self.acc_root / model, self.orig_root / model
        if not self.h_dir.exists():
            src = HARNESS_CFG / model
            if src.exists():
                copy_tree(src, self.h_dir)
            else:
                self.h_dir.mkdir(parents=True, exist_ok=True)
                (self.h_dir / "harness.md").write_text("", encoding="utf-8")
        for root in (self.acc_dir, self.orig_dir):
            if not root.exists():
                copy_tree(self.h_dir, root)
        self.state = driver.load_state(name)
        self.state.setdefault("steps", {})
        self.meta = self.state.setdefault("meta", {})

    @property
    def h_file(self):
        return self.h_dir / "harness.md"

    # ---- steps: every finished step is saved; a rerun skips it -------------------------------------------------------------
    def _save(self):
        driver.save_state(self.name, self.state)

    def _timebox(self):
        if self.timebox_min and time.time() - self.meta["started"] > self.timebox_min * 60:
            raise Stop("timebox")

    def _budget(self, editor):
        mine = [r for r in usage_rows(self.usage) if r.get("editor") == editor]
        cap = (self.limits.get("calls") or {}).get(editor)
        if cap is not None and len(mine) >= cap:
            raise Stop(f"budget: {editor} calls {len(mine)} of {cap}")
        cap_tok = self.limits.get("tokens")
        spent = sum(int(r.get("tokens_in") or 0) + int(r.get("tokens_out") or 0) for r in mine)
        if cap_tok is not None and spent >= cap_tok:
            raise Stop(f"budget: {editor} tokens {spent} of {cap_tok}")

    def step(self, key, fn):
        steps = self.state["steps"]
        if key in steps:
            return steps[key]["value"]
        self._timebox()
        value = fn()
        steps[key] = {"value": value, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._save()
        if self.crash_after == key:
            raise RuntimeError(f"crash injected after {key} (test hook)")
        return value

    def _run(self, task, root, arm, rep, out):
        rec = self.runner.run(task, root, self.model, arm, rep, out)
        if self.general is not None and task.get("set") == "general-v1":
            # the rubric judge scores writing/design/document/post answers right after the run: the record is final when the step ends
            rec = general.finish_record(rec, task, os.environ.get("GEVAL_ANSWER_DIR"), self.judge)
        return rec

    def _gen_runs(self, key, root, fp):
        """The general tasks of one set (fast, dev, test) under one harness root, every run its own step, cached by the fingerprint."""
        out = self.dir / "runs" / "general" / key / fp
        return [self._run_step(f"gen.{key}.{fp}.{t['id']}.{rep}", t, root, f"gen-{key}-{fp}", rep, out)
                for t in self.general[key] for rep in range(1, self.repeats + 1)]

    def _general_screen(self, tid, diff):
        """RSI1: a composed edit on the fast general set, against the working base: a gross drop of the general score refuses the edit
        before it becomes a version (the checkpoint's interval rule on the whole general dev part decides the version)."""
        root = self._trial(f"{tid}-composed", diff)
        fw, fc = executor.fingerprint(self.h_dir), executor.fingerprint(root / self.model)
        base = self._gen_runs("fast", self.h_root, fw)
        cand = self._gen_runs("fast", root, fc)
        gate = general.fast_gate(cand, base)
        return {"ok": bool(gate["ok"]), "gate": gate, "base": fw, "candidate": fc,
                "judge_failed": sum(1 for r in base + cand if r.get("judge_failed"))}

    def _run_step(self, key, task, root, label, rep, out):
        """One run is one step (RSIR1 D5): the time limit is checked between runs, and an interruption loses no finished run."""
        return self.step(key, lambda: self._run(task, root, label, rep, out))

    def _dev_all(self, root, fp):
        """The dev tasks under one harness, every run its own step; cached by the harness fingerprint (RSIR1 D3)."""
        out = self.dir / "runs" / "dev" / fp
        return [self._run_step(f"dev.{fp}.{t['id']}.{rep}", t, root, f"dev-{fp}", rep, out)
                for t in self.tasks["dev"] for rep in range(1, self.repeats + 1)]

    # ---- the cycle ---------------------------------------------------------------------------------------------------------
    def run_cycle(self):
        self.meta.setdefault("started", time.time())
        self.meta.update({"model": self.model, "fake": self.fake, "repeats": self.repeats, "limits": self.limits,
                          "timebox_min": self.timebox_min, "thresholds": self.thresholds, "allow_stub": self.allow_stub,
                          "checkpoint_every": self.checkpoint_every, "dev_scope": self.dev_scope,
                          "dev_ideas": sorted({t["family"].split("_", 1)[-1] for t in self.tasks["dev"]}),
                          "train": [t["id"] for t in self.tasks["train"]], "dev": [t["id"] for t in self.tasks["dev"]],
                          "dev_families": sorted({t["family"] for t in self.tasks["dev"]}),
                          "test": [t["id"] for t in self.tasks.get("test", [])], "status": "running"})
        if self.general is not None:
            self.meta["general"] = {k: [t["id"] for t in v] for k, v in self.general.items()}
        self._save()
        try:
            self.step("baseline-dev", lambda: self._dev_all(self.acc_root, executor.fingerprint(self.acc_dir)))
            train = self.tasks["train"]
            for i, task in enumerate(train):
                self._train_task(i, task)
                if self._checkpoint_due(i == len(train) - 1):
                    n = self._versions()
                    self.step(f"cp.{n:03d}", lambda n=n: self._checkpoint(n))
            if self.tasks.get("test"):
                self.step("final.test", self._final_test)
            status = "done"
        except Stop as stop:
            status = f"stopped: {stop}"
        self.meta.update(status=status, ended=time.time())
        self._save()
        return status

    def _versions(self):
        return sum(1 for k in self.state["steps"] if k.endswith(".version") and self.state["steps"][k]["value"].get("applied"))

    def _checked_versions(self):
        return max([v["value"].get("versions_at", 0) for k, v in self.state["steps"].items() if re.fullmatch(r"cp\.\d+", k)], default=0)

    def _checkpoint_due(self, last_task):
        pending = self._versions() - self._checked_versions()
        return pending > 0 and (last_task or (self.checkpoint_every and pending >= self.checkpoint_every))

    def _train_task(self, i, task):
        tid = f"t{i:02d}-{task['id']}"
        base = [self._run_step(f"{tid}.base.{rep}", task, self.h_root, "base", rep, self.dir / "runs" / tid)
                for rep in range(1, self.repeats + 1)]
        self.step(f"{tid}.pool", lambda: self._pool(i, tid))
        proposals = {}
        for role in ("invariant", "specific"):
            editor = editor_for(role, task, base, i, self.editors)
            proposals[role] = self.step(f"{tid}.propose.{role}", lambda role=role, editor=editor: self._propose(tid, role, editor))
        diffs = {r: p["diff"] for r, p in proposals.items() if p["diff"]}
        evidence, composed = {}, {"diff": "", "keep": True}
        if not diffs:
            outcome = "no candidate"
        else:
            measured = {r: self.step(f"{tid}.ev.{r}", lambda r=r, d=d: self._task_measure(tid, r, d, task, base))
                        for r, d in diffs.items()}
            for r, m in measured.items():
                if not m.get("applies"):   # RSIR1 D7: a candidate that was not measured does not reach the composer
                    (self.dir / "candidates" / tid / f"{r}.diff").unlink(missing_ok=True)
            applied = [r for r, m in measured.items() if m.get("applies")]
            if not applied:
                outcome = "no measured candidate"
            else:
                evidence = self._evidence({r: measured[r] for r in applied})
                composed = self.step(f"{tid}.compose", lambda: self._compose(tid, evidence))
                if not composed["keep"] and self.general is not None and self.general.get("fast"):
                    screen = self.step(f"{tid}.general", lambda: self._general_screen(tid, composed["diff"]))
                    if not screen["ok"]:
                        composed = dict(composed, keep=True, refused="general regression", general=screen["gate"])
                if composed["keep"]:
                    outcome = "KEEP_BASE" + (f" ({composed['refused']})" if composed.get("refused") else "")
                else:
                    self.step(f"{tid}.version", lambda: self._new_version(tid, composed["diff"]))
                    outcome = "version"
        self.step(f"{tid}.reflect", lambda: self._reflect(tid, task, outcome, evidence, composed))

    # ---- the pool: base runs only (RSIR1 D2) ---------------------------------------------------------------------------------
    def _pool_dir(self, tid):
        return self.dir / "pools" / tid / "trajectories"

    def _pool(self, i, tid):
        """The trajectories an editor may read: the base runs of the recent train tasks, each made under the harness of its own time.
        RSIL0e E1: every repeat of a base task goes in, not only repeat 1; a task whose repeats disagree is marked. The pool makes no runs
        of its own. The chosen ones are written as files; the state records which ones entered."""
        pool_idx = range(max(0, i - POOL_TASKS + 1), i + 1)
        entries, disagree = [], []
        for j in pool_idx:
            t = self.tasks["train"][j]
            tj = f"t{j:02d}-{t['id']}"
            runs = {}
            for rep in range(1, self.repeats + 1):
                rec = self.state["steps"].get(f"{tj}.base.{rep}", {}).get("value")
                log = self.dir / "runs" / tj / f"{t['id']}__base__{rep}.agent.log"
                if rec and log.exists():
                    runs[rep] = (rec, log)
            if len({bool(rec.get("resolved")) for rec, _ in runs.values()}) > 1:
                disagree.append(t["id"])
            for rep, (rec, log) in runs.items():
                others = [r for n, (r, _) in runs.items() if n != rep]
                entries.append(trajectory.entry(rec, t, log, repeats=self.repeats, others=others))
        sel = trajectory.select(entries)
        trajectory.write_pool(self._pool_dir(tid), sel["chosen"])
        return {"pool_tasks": [self.tasks["train"][j]["id"] for j in pool_idx], "chosen": [e["id"] for e in sel["chosen"]],
                "fail_signs": {e["id"]: e["fail"] for e in sel["chosen"] if e["fail"]}, "chars": sel["chars"],
                "cut": sel["cut"], "left_out": sel["left_out"], "disagree": disagree, "own_runs": 0}

    # ---- the editors -----------------------------------------------------------------------------------------------------------
    def _library_view(self):
        """What an editor may read from the library (RSIR1 D9): EXPERIENCE.md, INDEX.md and the accepted checkpoints' folders. The versions
        and the archive stay out."""
        view = self.dir / "library_view"
        if view.exists():
            shutil.rmtree(view)
        if not self.library.exists():
            return None
        view.mkdir(parents=True)
        for name in ("EXPERIENCE.md", "INDEX.md"):
            if (self.library / name).exists():
                shutil.copy2(self.library / name, view / name)
        harnesses = self.library / "harnesses"
        for entry in sorted(harnesses.glob("cp*")) if harnesses.exists() else []:
            copy_tree(entry, view / "harnesses" / entry.name)
        return view

    def _growth_of(self, root):
        orig = (self.orig_dir / "harness.md").read_text(encoding="utf-8")
        return growth((Path(root) / self.model / "harness.md").read_text(encoding="utf-8"), orig)

    def _propose(self, tid, role, editor):
        self._budget(editor)
        os.environ["RSI_FAKE_BASE"] = str(self.h_file)          # read by the stub editor of --fake only
        cand_file = self.dir / "candidates" / tid / f"{role}.diff"
        try:
            diff = driver.propose(self.name, role, self.h_dir, None, usage_log=self.usage, editor=editor, model=self.model,
                                  library_dir=self._library_view(), cand_dir=self.dir / "candidates" / tid,
                                  trajectories=self._pool_dir(tid))
        except sandbox.LeakError as err:      # RSIL0d P4: a leak in the editor's folder is a refusal of the surface, not a crash
            cand_file.unlink(missing_ok=True)
            return {"editor": editor, "diff": "", "refused": "surface", "reason": str(err)[:200]}
        if not diff:
            return {"editor": editor, "diff": "", "refused": None}
        if not surface_ok(diff):
            cand_file.unlink(missing_ok=True)
            return {"editor": editor, "diff": "", "refused": "surface"}
        size = surface_size(diff)
        root = self._trial(f"{tid}-{role}", diff)
        if not size["ok"] or root is None:
            cand_file.unlink(missing_ok=True)   # rejected unread, as the card says: it does not reach the composer
            return {"editor": editor, "diff": "", "refused": "size" if not size["ok"] else "does not apply", "size": size}
        growth_info = self._growth_of(root)
        if not growth_info["ok"]:
            cand_file.unlink(missing_ok=True)
            return {"editor": editor, "diff": "", "refused": "budget", "growth": growth_info}
        return {"editor": editor, "diff": diff, "refused": None, "size": size, "growth": growth_info}

    def _trial(self, label, diff):
        """A copy of the working harness with the diff applied: a harness root (<root>/<model>/...) for the runs."""
        root = self.dir / "trials" / label
        if not (root / self.model / "harness.md").exists():
            if root.exists():
                shutil.rmtree(root)
            (root / self.model).mkdir(parents=True)
            copy_tree(self.h_dir, root / self.model)
            if not driver.apply_diff(root / self.model, diff):
                shutil.rmtree(root)
                return None
        return root

    # ---- the task level ----------------------------------------------------------------------------------------------------
    def _task_measure(self, tid, role, diff, task, base):
        """The candidate and the working base on THIS train task, repeats each (evidence for the composer; no interval rule here)."""
        root = self._trial(f"{tid}-{role}", diff)
        if root is None:
            return {"applies": False}
        cand = [self._run_step(f"{tid}.ev.{role}.{rep}", task, root, f"{tid}-{role}", rep, self.dir / "runs" / tid / role)
                for rep in range(1, self.repeats + 1)]
        verdict = accept.decide(cand, base, thresholds=self.thresholds, allow_stub=self.allow_stub)
        return {"applies": True, "level": "task", "verdict": verdict, "families": family_means(cand, base),
                "runs": paired_runs(cand, base)}

    @staticmethod
    def _evidence(measured):
        """RSIP1 §4: {"invariant": accept.decide, "specific": accept.decide, "families": [...]}; a role that was not measured is None.
        Each verdict carries its runs side by side and its level (RSIR1 D8)."""
        ev = {"invariant": None, "specific": None}
        fams = {}
        for role, m in measured.items():
            if not m.get("applies"):
                continue
            ev[role] = dict(m["verdict"], level=m["level"], runs=m["runs"])
            for fam, d in m["families"].items():
                fams.setdefault(fam, {"family": fam, "invariant": None, "specific": None})[role] = d
        ev["families"] = [fams[f] for f in sorted(fams)]
        return ev

    def _compose(self, tid, evidence):
        self._budget("fable")
        os.environ["RSI_FAKE_BASE"] = str(self.h_file)
        try:
            diff = driver.compose(self.name, self.h_dir, evidence, usage_log=self.usage, model=self.model,
                                  library_dir=self._library_view(), cand_dir=self.dir / "candidates" / tid)
        except sandbox.LeakError:
            return {"diff": "", "keep": True, "refused": "surface"}
        if not diff:
            return {"diff": "", "keep": True}
        if not surface_ok(diff):
            return {"diff": "", "keep": True, "refused": "surface"}
        if not surface_size(diff)["ok"]:
            return {"diff": "", "keep": True, "refused": "size"}
        root = self._trial(f"{tid}-composed", diff)
        if root is None:
            return {"diff": "", "keep": True, "refused": "does not apply"}
        growth_info = self._growth_of(root)
        if not growth_info["ok"]:           # RSIR1 D6: the composed version is checked too
            return {"diff": "", "keep": True, "refused": "budget", "growth": growth_info}
        return {"diff": diff, "keep": False, "growth": growth_info}

    def _new_version(self, tid, diff):
        """The composed diff becomes the working harness (the next version), and is archived in the library's versions folder."""
        root = self._trial(f"{tid}-composed", diff)
        if root is None:
            return {"applied": False, "reason": "diff does not apply"}
        copy_tree(root / self.model, self.h_dir)
        archive = self.library / "versions" / tid
        copy_tree(self.h_dir, archive)
        return {"applied": True, "fingerprint": executor.fingerprint(self.h_dir), "archive": str(archive)}

    def _reflect(self, tid, task, outcome, evidence, composed):
        """The reflector sees the round (RSIP1 §4: `# round/`). When nothing was measured, one line is written and no call is made (P5)."""
        self.library.mkdir(parents=True, exist_ok=True)
        exp = self.library / "EXPERIENCE.md"
        if outcome in ("no candidate", "no measured candidate"):
            with open(exp, "a", encoding="utf-8") as f:
                f.write(f"\n## {tid} — {outcome}\n- nothing to learn from this task: {outcome}\n")
            return {"ok": True, "lines": 1, "called": False}
        self._budget("sonnet")
        os.environ["RSI_FAKE_BASE"] = str(self.h_file)
        cand = self.dir / "candidates" / tid
        notes = [(cand / f"{r}.reply.txt").read_text(encoding="utf-8") for r in ("invariant", "specific")
                 if (cand / f"{r}.reply.txt").exists()]
        composed_note = (cand / "composed.reply.txt").read_text(encoding="utf-8") if (cand / "composed.reply.txt").exists() else ""
        round_text = "\n".join(["## base (harness/)", self.h_file.read_text(encoding="utf-8"), "## candidates (diffs with their notes)",
                                "\n\n".join(notes) or "(none)", "## composition", composed_note or "(no composition)", "## verdicts",
                                json.dumps({"task": task["id"], "outcome": outcome, "evidence": evidence,
                                            "composer": {k: v for k, v in composed.items() if k != "diff"}}, ensure_ascii=False, indent=1)])
        library_text = exp.read_text(encoding="utf-8") if exp.exists() else ""
        prompt = "\n".join([(driver.PROMPTS / "reflector.md").read_text(encoding="utf-8"), f"# model: {self.model}", "# role: reflector",
                            "", "# round/", round_text] + (["# library/", library_text] if library_text else []))
        scratch = Path(tempfile.mkdtemp(prefix="rsil-reflect-", dir="/dev/shm"))
        try:
            reply = editors.call("sonnet", "reflector", prompt, cwd=scratch, usage_log=self.usage)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        keep = outcome.startswith("KEEP_BASE")
        text = reply.text.strip() if reply.ok else ""
        if not reply.ok:
            lines = [f"(no entry: {reply.error[:120]})"]
        elif keep and (not text or text.upper().startswith("NO_ENTRY")):
            lines = ["- KEEP_BASE: no new lesson"]          # RSIL0e: a KEEP_BASE round without a lesson is one line, not an entry
        else:
            body = text.splitlines()
            while body and (not body[0].strip() or body[0].startswith("# ")):   # RSIL0e E6: the driver writes the round heading itself
                body.pop(0)
            lines = body[:12 if keep else 40] or ["- no new lesson"]
        with open(exp, "a", encoding="utf-8") as f:
            f.write(f"\n## {tid} — {outcome}\n" + "\n".join(lines) + "\n")
        return {"ok": reply.ok, "lines": len(lines), "called": True}

    # ---- the checkpoints -------------------------------------------------------------------------------------------------
    def _checkpoint(self, n):
        """The working harness against the last accepted one on the dev tasks (paired, DATASET.md §5). Accepted: it becomes the accepted
        harness. Refused: the working harness goes back to the accepted one, and the numbers go to the library."""
        fw, fa = executor.fingerprint(self.h_dir), executor.fingerprint(self.acc_dir)
        if fw == fa:
            return {"versions_at": n, "accepted": False, "same": True, "fingerprint": fa}
        cand = self._dev_all(self.h_root, fw)
        base = self._dev_all(self.acc_root, fa)
        verdict = accept.decide(cand, base, thresholds=self.thresholds, allow_stub=self.allow_stub)
        general_verdict = None
        if verdict["accept"] and self.general is not None and self.general.get("dev"):
            # the code verdict allows it; the general dev part must not fall beyond noise (rsi/accept_general.py); only now it is run
            gcand, gbase = self._gen_runs("dev", self.h_root, fw), self._gen_runs("dev", self.acc_root, fa)
            general_verdict = accept_general.decide_with_general(cand, base, general.usable(gcand), general.usable(gbase), model_kind="code",
                                                                 thresholds=self.thresholds, allow_stub=self.allow_stub)
            general_verdict["judge_failed"] = sum(1 for r in gcand + gbase if r.get("judge_failed"))
            verdict = dict(verdict, accept=bool(general_verdict["accept"]), code_accept=True)
        if verdict["accept"]:
            copy_tree(self.h_dir, self.acc_dir)
            copy_tree(self.acc_dir, self.library / "harnesses" / f"cp{n:03d}")
            self._index()
            outcome = f"checkpoint {n}: accepted"
        else:
            copy_tree(self.acc_dir, self.h_dir)
            outcome = f"checkpoint {n}: refused, back to the accepted version" + (
                " (general tasks)" if general_verdict is not None and verdict.get("code_accept") else "")
        self.library.mkdir(parents=True, exist_ok=True)
        with open(self.library / "EXPERIENCE.md", "a", encoding="utf-8") as f:
            f.write(f"\n## cp{n:03d} — {outcome}\nscore {verdict['score']['mean']:+.3f} {verdict['score']['ci95']}, "
                    f"resolved {verdict['resolved']['mean']:+.3f} {verdict['resolved']['ci95']}, "
                    f"time x{verdict['time_ratio']}, tokens x{verdict['tokens_ratio']}, "
                    f"families {json.dumps(verdict.get('family_regressions', {}), ensure_ascii=False)}\n")
            if general_verdict is not None:
                g = general_verdict["guard"]
                f.write(f"general dev: score {g['score']['mean']:+.3f} {g['score']['ci95']}, pairs {g['pairs']}, noise {g['noise']} "
                        f"({'ok' if g['ok'] else 'FALLS BEYOND NOISE'}; {g['noise_source'][:40]})\n")
        return {"versions_at": n, "accepted": bool(verdict["accept"]), "outcome": outcome, "verdict": verdict, "general": general_verdict,
                "fingerprint": executor.fingerprint(self.acc_dir)}

    def _index(self):
        harnesses = self.library / "harnesses"
        rows = sorted(p for p in harnesses.iterdir() if p.is_dir()) if harnesses.exists() else []
        lines = [f"# Library — {self.model}", "", "Each entry is a checkpoint-accepted harness (the whole folder). EXPERIENCE.md says why.", ""]
        for p in rows:
            skills = sorted(s.name for s in (p / "skills").glob("*") if (s / "SKILL.md").exists()) if (p / "skills").exists() else []
            lines.append(f"- `harnesses/{p.name}/harness.md`" + (f" (skills: {', '.join(skills)})" if skills else ""))
        self.library.mkdir(parents=True, exist_ok=True)
        (self.library / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- the final test: once, reported only --------------------------------------------------------------------------------
    def _final_test(self):
        test = self.tasks["test"]
        fo, fa = executor.fingerprint(self.orig_dir), executor.fingerprint(self.acc_dir)
        base = [self._run_step(f"final.{fo}.{t['id']}.{rep}", t, self.orig_root, f"orig-{fo}", rep, self.dir / "runs" / "test" / fo)
                for t in test for rep in range(1, self.repeats + 1)]
        cand = [self._run_step(f"final.{fa}.{t['id']}.{rep}", t, self.acc_root, f"acc-{fa}", rep, self.dir / "runs" / "test" / fa)
                for t in test for rep in range(1, self.repeats + 1)]
        verdict = accept.decide(cand, base, thresholds=self.thresholds, allow_stub=self.allow_stub)
        out = {"tasks": [t["id"] for t in test], "repeats": self.repeats, "original": fo, "accepted": fa, "verdict": verdict,
               "note": "reported only: the test tasks are not used for any decision"}
        if self.general is not None and self.general.get("test"):
            gbase, gcand = self._gen_runs("test", self.orig_root, fo), self._gen_runs("test", self.acc_root, fa)
            out["general"] = accept_general.decide_with_general(cand, base, general.usable(gcand), general.usable(gbase), model_kind="code",
                                                                thresholds=self.thresholds, allow_stub=self.allow_stub)
            out["general_tasks"] = [t["id"] for t in self.general["test"]]
        return out


# ---- the command line -----------------------------------------------------------------------------------------------------
def parse_limits(calls_text, max_tokens, train_count=0):
    """Per editor calls. Empty: the night's defaults (RSIR1 D10, RSIL0e E3): opus and sol 2·train+2 each, fable 2·train+2, sonnet
    train+2."""
    calls = {}
    if calls_text:
        for part in filter(None, calls_text.split(",")):
            k, v = part.split("=")
            calls[k.strip()] = int(v)
    else:
        calls = {"opus": 2 * train_count + 2, "sol": 2 * train_count + 2, "fable": 2 * train_count + 2,   # RSIL0e E3: Fable takes more
                 "sonnet": train_count + 2}                                                                # specifics
    return {"calls": calls, "tokens": max_tokens}


def setup_stub_editor(workdir):
    """The `claude` the editors call in --fake: a wrapper that runs the stub, so no model is called."""
    stub_dir = Path(workdir) / "stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    wrapper = stub_dir / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec python3 -B {STUB_EDITOR} \"$@\"\n")
    wrapper.chmod(0o755)
    editors.CLAUDE_BIN = str(wrapper)
    codex = stub_dir / "codex"               # RSI1: Sol in --fake is the same stub, called the way codex is called
    codex.write_text(wrapper.read_text())
    codex.chmod(0o755)
    editors.CODEX_BIN = str(codex)


def barid(*args):
    return subprocess.run(["python3", BARID, "--board", BOARD, *args], capture_output=True, text=True)


def fake_judge(task_dir, answer_text, image=None):
    """--fake: no Claude call for the rubric judge; every answer gets the same middle score (the numbers mean nothing in a fake run)."""
    return {"score": 0.5, "criteria": [1]}


def run_cli(args):
    from . import accept as _accept
    from . import server_argv
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 (reports/rsit1/DATASET.md §5)")
    try:
        thresholds, allow_stub = _accept.load_thresholds(), False
    except FileNotFoundError as err:
        if not args.fake:
            raise SystemExit(f"refused: {err}")
        thresholds, allow_stub = _accept.STUB, True          # the STUB object itself: accept.decide marks its verdicts as stub
    workdir = driver.RUNS / args.run
    tasks = select_tasks(args.train, args.dev, args.test, dev_scope=args.dev_scope)
    skip = set(args.skip_task)
    skipped = sorted(t["id"] for t in tasks["dev"] if t["id"] in skip)
    tasks["dev"] = [t for t in tasks["dev"] if t["id"] not in skip]
    if (not tasks["train"] or not tasks["dev"]) and getattr(args, "step", "pilot") != "lhcal":
        raise SystemExit("no train or dev tasks found for the split")
    general_tasks = None
    if getattr(args, "general", "off") == "on":
        general_tasks = general.select_general()
        if not general_tasks["fast"] or not general_tasks["dev"]:
            raise SystemExit("--general on: no general dev tasks found for the split")
        # the answers of the judged tasks and the simulated user's cache are shared by every run of this run (all arms, all nights)
        os.environ.setdefault("GEVAL_ANSWER_DIR", str(workdir / "answers"))
    os.environ.setdefault("AGENTIC_EVAL_SIMUSER_CACHE", str(workdir / "simuser-cache"))
    limits = parse_limits(args.max_calls, args.max_tokens, len(tasks["train"]))
    editors = tuple(e.strip() for e in getattr(args, "editors", "opus").split(",") if e.strip())
    if not editors or any(e not in ("opus", "opus-high", "sol") for e in editors):
        raise SystemExit(f"--editors: a comma list of opus, opus-high, sol; got {args.editors!r}")
    if args.fake:
        setup_stub_editor(workdir)
        stand = executor.FakeStand(workdir / "fake")
    elif getattr(args, "stand", "server") == "gateway":
        from .gateway_stand import GatewayStand
        if not args.gateway_commit:
            raise SystemExit("the gateway stand needs --gateway-commit (the commit its worktree is pinned to)")
        stand = GatewayStand(workdir / "gw", executor.MODEL_IDS[args.model], args.gateway_worktree, args.gateway_commit, effort=args.effort)
    else:
        argv_file = Path(args.server_argv) if args.server_argv else workdir / "server-argv.json"
        if not args.server_argv:
            argv_file.parent.mkdir(parents=True, exist_ok=True)
            argv_file.write_text(json.dumps(server_argv.build(executor.MODEL_IDS[args.model])), encoding="utf-8")   # the gateway's build_args
        stand = executor.GpuStand(argv_file, workdir / "gpu")
    if not args.fake:
        held = barid("hold", "gpu", "--task", args.gpu_task, "--wait", "600", "--by", args.agent)
        if held.returncode != 0:
            raise SystemExit(f"GPU not held: {held.stdout.strip() or held.stderr.strip()}")
    try:
        with stand:
            runner = executor.Runner(stand, workdir / "scratch", context=getattr(args, "engine_context", None))
            common = dict(thresholds=thresholds, allow_stub=allow_stub, repeats=args.repeats, limits=limits, timebox_min=args.timebox_min,
                          fake=args.fake, crash_after=args.crash_after, checkpoint_every=args.checkpoint_every, dev_scope=args.dev_scope,
                          editors=editors, general=general_tasks, judge=(fake_judge if args.fake else None))
            if getattr(args, "step", "pilot") == "lhcal":       # LHSET1: the long-horizon calibration, no editors, no decision
                from . import rsi2
                summary = rsi2.lh_calibration(args.run, args.model, runner, part=args.lh_part, n=args.lh_n, repeats=args.repeats)
                print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))
                return 0
            if getattr(args, "step", "pilot") == "cycle2":      # RSI2D: the cause-driven night (rsi/rsi2.py); `pilot` stays as it was
                from . import rsi2
                pilot = rsi2.Cycle2(args.run, args.model, tasks, runner, max_candidates=args.max_candidates, min_room=args.min_room,
                                    prior_runs=[x for x in args.prior_runs.split(",") if x], max_growth=args.max_growth,
                                    cause_editor=args.cause_editor, mechanisms=[x for x in args.mechanisms.split(",") if x],
                                    safety_ru=rsi2.ru_safety_tasks(args.safety_ru), screen_reps=args.screen_reps, lh_final=args.lh_final, **common)
            else:
                pilot = Pilot(args.run, args.model, tasks, runner, **common)
            if skipped:   # RSIL1-b: the reason and the tasks are kept in the state and the report
                pilot.meta["skipped"] = {"tasks": skipped, "reason": args.skip_reason}
            if hasattr(stand, "commit"):   # RSI1: the gateway stand's pin and effort, kept in the state for the morning report
                pilot.meta["stand"] = {"kind": "gateway", "commit": getattr(stand, "head", stand.commit), "effort": stand.effort,
                                       "model_id": stand.model_id, "sampling": stand.sampling, "pins": stand.pins}
            runner.discarded = pilot.meta.setdefault("discarded", [])   # a resumed run keeps the earlier nights' list
            status = pilot.run_cycle()
    finally:
        if not args.fake:
            barid("drop", "gpu", "--task", args.gpu_task, "--by", args.agent)
    print(f"pilot {args.run}: {status}")
    print(morning(args.run))
    return 0


# ---- the morning report -----------------------------------------------------------------------------------------------------
RUN_CLASSES = [("base (задача, до предложений)", r"^t\d\d-.*\.base\.\d+$"),
               ("general (экран правки, dev, test)", r"^gen\."),
               ("кандидаты (замер на задаче)", r"^t\d\d-.*\.ev\.\w+\.\d+$"),
               ("контрольные точки и базовый dev", r"^dev\.[0-9a-f]+\.[^.]+\.\d+$"),
               ("финальный тест", r"^final\.")]


def stand_lines(meta):
    """RSI1: the gateway stand's pin and effort, and the runs not counted («200 и корректно» failed, the stand died)."""
    out = []
    st = meta.get("stand")
    if st:
        out.append(f"- стенд: шлюз {str(st.get('commit'))[:9]}, {st.get('model_id')}, effort {st.get('effort')}, "
                   f"сэмплинг {json.dumps(st.get('sampling'), ensure_ascii=False)}")
    dropped = meta.get("discarded") or []
    if dropped or st:
        client = sum(1 for d in dropped if (d.get("client") or {}).get("bad_total"))
        out.append(f"- не засчитано прогонов: {len(dropped)} (из них клиент получил ошибку или недописанный ответ: {client})")
    return out


def morning(name):
    state = driver.load_state(name)
    meta, steps = state.get("meta", {}), state.get("steps", {})
    rows = usage_rows(driver.RUNS / name / "usage.jsonl")
    per = {}
    for r in rows:
        e = per.setdefault(r.get("editor"), {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "errors": 0})
        e["calls"] += 1
        e["tokens_in"] += int(r.get("tokens_in") or 0)
        e["tokens_out"] += int(r.get("tokens_out") or 0)
        e["cost_usd"] += float(r.get("cost_usd") or 0.0)
        e["errors"] += 0 if r.get("ok") else 1
    elapsed = ((meta.get("ended") or time.time()) - meta.get("started", time.time())) / 60
    dev_fams = len(meta.get("dev_families") or [])
    out = [f"# Утренний отчёт пилота {name}", "",
           f"- модель: {meta.get('model')}, режим: {'--fake (без GPU, заглушки редакторов)' if meta.get('fake') else 'настоящий'}",
           f"- статус: {meta.get('status')}, время: {elapsed:.1f} мин",
           f"- пороги: {json.dumps(meta.get('thresholds'), ensure_ascii=False)}",
           f"- dev ({meta.get('dev_scope', 'subset')}, идеи {', '.join(meta.get('dev_ideas') or [])}): "
           f"{len(meta.get('dev') or [])} задач из {dev_fams} семейств"
           + ("" if dev_fams >= 2 else " — интервал по семействам не определён, решение не годится"),
           *stand_lines(meta), "",
           "## Задачи (уровень задачи)", "", "| задача | пул траекторий | кандидаты (предложено) | замер на задаче | композитор | итог |",
           "|---|---|---|---|---|---|"]
    tids = sorted({k.split(".")[0] for k in steps if k.split(".")[0][:1] == "t" and "-" in k.split(".")[0]})
    for tid in tids:
        pool = steps.get(f"{tid}.pool", {}).get("value") or {}
        props = [steps[k]["value"] for k in (f"{tid}.propose.invariant", f"{tid}.propose.specific") if k in steps]
        measures = []
        for k in sorted(k for k in steps if k.startswith(f"{tid}.ev.") and k.count(".") == 2):
            v = steps[k]["value"]
            if v.get("applies"):
                measures.append(f"{k.split('.')[-1]}: score {v['verdict']['score']['mean']:+.3f}")
        comp = steps.get(f"{tid}.compose", {}).get("value") or {}
        reflected = steps.get(f"{tid}.reflect", {}).get("value") or {}
        if comp.get("keep"):
            outcome = "KEEP_BASE" + (f" ({comp['refused']})" if comp.get("refused") else "")
        elif comp.get("diff"):
            outcome = "version"
        else:
            outcome = "no measured candidate" if f"{tid}.ev.invariant" in steps or f"{tid}.ev.specific" in steps else "no candidate"
        pool_txt = f"{len(pool.get('chosen', []))} из {len(pool.get('pool_tasks', []))} задач, {pool.get('chars', 0)} симв." \
            + (", обрезано" if pool.get("cut") else "")
        out.append(f"| {tid} | {pool_txt} | {sum(1 for p in props if p.get('diff'))} из {len(props)} | "
                   f"{'; '.join(measures) or '-'} | {'KEEP' if comp.get('keep') else ('diff' if comp.get('diff') else '-')} | {outcome} |")
    out += ["", "## Контрольные точки (dev, DATASET §5)", "", "| точка | версий | итог | score | resolved | время, токены |", "|---|---|---|---|---|---|"]
    for k in sorted(k for k in steps if re.fullmatch(r"cp\.\d+", k)):
        v = steps[k]["value"]
        if v.get("same"):
            out.append(f"| {k} | - | та же версия, что принятая | - | - | - |")
        else:
            sc, rs, vr = v["verdict"]["score"], v["verdict"]["resolved"], v["verdict"]
            out.append(f"| {k} | {v['versions_at']} | {v['outcome']} | {sc['mean']:+.3f} {sc['ci95']} | {rs['mean']:+.3f} {rs['ci95']} | "
                       f"x{vr['time_ratio']} (лимит {vr['time_limit']}), x{vr['tokens_ratio']} |")
    if meta.get("general"):
        g = meta["general"]
        out += ["", "## General-задачи (accept_general: правка не должна ронять общие задачи)", "",
                f"- быстрый набор ({len(g.get('fast', []))}): {', '.join(g.get('fast', []))}; полный dev: {len(g.get('dev', []))}, test: {len(g.get('test', []))}",
                "", "| где | итог | score (ДИ) | пар | шум |", "|---|---|---|---|---|"]
        for k in sorted(k for k in steps if k.endswith(".general") and k.startswith("t")):
            v = steps[k]["value"]
            gt = v["gate"]
            out.append(f"| правка {k.split('.')[0]} (экран) | {'прошла' if v['ok'] else 'ОТКЛОНЕНА: общий score упал'} | "
                       f"{gt['score']['mean']:+.3f} {gt['score']['ci95']} | {gt['pairs']} | {gt['noise']} |" if gt.get("score") else
                       f"| правка {k.split('.')[0]} (экран) | нет пар | - | 0 | - |")
        for k in sorted(k for k in steps if re.fullmatch(r"cp\.\d+", k)):
            gv = (steps[k]["value"] or {}).get("general")
            if gv:
                gd = gv["guard"]
                out.append(f"| {k} (весь general dev) | {'не падает' if gd['ok'] else 'ПАДАЕТ ВЫШЕ ШУМА'} | "
                           f"{gd['score']['mean']:+.3f} {gd['score']['ci95']} | {gd['pairs']} | {gd['noise']} |")
        out.append(f"- шум NOISE: {accept_general.NOISE_SOURCE}")
    out += ["", "## Доля прогонов по статьям (RSIR1: бюджет ночи)", "", "| класс | прогонов | медиана, с | доля |", "|---|---|---|---|"]
    counts = {}
    for label, pattern in RUN_CLASSES:
        secs = [float(v["value"].get("seconds") or 0) for k, v in steps.items() if re.search(pattern, k) and isinstance(v.get("value"), dict)]
        counts[label] = secs
    total = sum(len(v) for v in counts.values()) or 1
    for label, secs in counts.items():
        med = sorted(secs)[len(secs) // 2] if secs else 0
        out.append(f"| {label} | {len(secs)} | {med:.0f} | {100 * len(secs) / total:.0f} % |")
    out.append(f"| прогонов пула (собственных) | {sum(1 for k in steps if k.startswith('traj.'))} | - | 0 % |")
    fin = steps.get("final.test", {}).get("value")
    out += ["", "## Финальный тест (один раз, только отчёт)", ""]
    if fin:
        out.append(f"- test-задачи: {', '.join(fin['tasks'])}, повторов {fin['repeats']}; score {fin['verdict']['score']['mean']:+.3f} "
                   f"{fin['verdict']['score']['ci95']}, resolved {fin['verdict']['resolved']['mean']:+.3f} {fin['verdict']['resolved']['ci95']}")
    else:
        out.append("- не запускался (цикл не дошёл до конца или нет test-задач)")
    out += ["", "## Расход по редакторам (usage.jsonl этого пилота)", "", "| редактор | вызовов | вход, токены | выход, токены | USD | ошибок |",
            "|---|---|---|---|---|---|"]
    for e, v in sorted(per.items(), key=lambda kv: str(kv[0])):
        out.append(f"| {e} | {v['calls']} | {v['tokens_in']} | {v['tokens_out']} | {v['cost_usd']:.4f} | {v['errors']} |")
    lib = driver.RUNS / name / "library" if meta.get("fake") else REAL_LIBRARY / str(meta.get("model"))
    out += ["", f"Библиотека: {lib} (INDEX.md, EXPERIENCE.md, harnesses/, versions/)", ""]
    text = "\n".join(out)
    (driver.RUNS / name / "MORNING.md").write_text(text, encoding="utf-8")
    return text
