"""Lifelong level 2, the gate (LL4b): one night measures ONE candidate rule of the person's personal layer against the current layer on
the person's own control tasks, with the pre-registered rule of RSI2 (decide2), and writes the verdict back into the knowledge store.

What is different from the RSI2 night (rsi2.Cycle2): nothing is mined and no editor writes anything. The candidate is a `queued` rule
in the store (mcp/rules.py of ShuraCode wrote it from the person's confirmed signals), the tasks are the `control_task` rows (LL3a), the
decision is decide2 + safety_ok + the cost limits (decisive here, because one rule IS the version), and the result is a `gate_run` row,
the rule's status (`passed` or `rejected`, with the numbers in its notes), a journal line and a line in the layer's EXPERIENCE.md. The
person enables a passed rule in the terminal (DESIGN §11 p. 1); the gate never switches anything on.

Ablation (§4.7): with `ablate=<key>` the candidate is the current layer WITHOUT one enabled rule; "the layer is not worse without it" is
recorded in the rule's notes (`ablation: no_benefit`) for the person to retire it. Nothing is retired by code.

Stand: `runner.run(task, root, model, arm, rep, out)` as everywhere in rsi (executor.Runner on the gateway stand, a fake runner here);
every record must say the harness text reached the model (`harness_delivery.delivered`, RSI2D), or the run is discarded and the night
stops. The harness root given to the runner holds <model>/harness.md = the product's base harness + the personal layer (+ the candidate),
which is what the two system messages of plugin/model-harness.ts amount to.

    cd bench/agentic-eval && python3 -m rsi.driver --run ll-night-1 lifelong --model tiel --fake --db /path/lifelong.sqlite --skills-dir /path/skills
"""
import importlib.util
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path

from . import accept, driver, executor, pilot, rsi2

GATEWAY = Path(__file__).resolve().parents[3] / "gateway"
DEFAULT_DB = Path.home() / ".local/share/shura/lifelong.sqlite"
DEFAULT_SKILLS = Path.home() / ".local/share/shura/skills"
GROWTH_MAX_MEAN = 0.9      # §4.4: a task with room has a base mean <= this
MIN_GROWTH = 4             # fewer tasks with room: no measurement tonight (the bound would be meaningless)
MIN_FAMILIES = 3           # the rule's units are families (repositories): fewer and the t bound is not a measurement
MAX_ENABLED = 12           # §4.7, the same numbers as mcp/rules.py: a candidate that could not be enabled is not measured
TOKEN_BUDGET = 600
CHARS_PER_TOKEN, TOKEN_MARGIN = 3.5, 1.15


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def store_schema():
    """The knowledge store's DDL (gateway/lifelong_store.py), so a night on a fresh file sees the same tables as the store and the MCP."""
    f = GATEWAY / "lifelong_store.py"
    spec = importlib.util.spec_from_file_location("lifelong_store", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCHEMA


RULE_CASE_DDL = "CREATE TABLE IF NOT EXISTS rule_case(id INTEGER PRIMARY KEY, rule_id INTEGER REFERENCES rule(id), session TEXT, ts TEXT, why TEXT, ref TEXT);"


def est_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN * TOKEN_MARGIN + 0.999)


class RuleStore:
    """The gate's view of the knowledge store: candidates, enabled rules, control tasks, gate runs. Same conventions as mcp/rules.py
    (the rule's key, model, when, do, why and gate numbers live in rule.notes as JSON)."""

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
        fresh = not self.path.exists()
        self.db = sqlite3.connect(str(self.path), isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=10000")
        if fresh:
            self.db.execute("PRAGMA page_size=1024")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(store_schema() + RULE_CASE_DDL)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def rules(self, model, statuses):
        out = []
        for r in self.db.execute("SELECT * FROM rule WHERE kind IN ('rule','procedure') AND status IN (%s) ORDER BY id" % ",".join("?" * len(statuses)), tuple(statuses)):
            try:
                notes = json.loads(r["notes"] or "{}")
            except ValueError:
                continue
            if notes.get("model") == model and notes.get("key"):
                out.append({**dict(r), "notes": notes})
        return out

    def control_tasks(self, split="work"):
        """control_task rows as task dicts for the runner: id ct<id>, family, kind, prompt, dir (the snapshot), check, timeout."""
        out = []
        for r in self.db.execute("SELECT * FROM control_task WHERE split=? ORDER BY id", (split,)):
            out.append({"id": f"ct{r['id']}", "ct_id": r["id"], "family": r["family"] or f"ct{r['id']}", "kind": r["check_kind"] or "implement",
                        "prompt": r["prompt"], "dir": r["snapshot_path"], "repo": r["repo"], "commit": r["commit"], "check_kind": r["check_kind"],
                        "check_ref": r["check_ref"], "timeout_s": r["timeout_s"], "solved_before": r["solved_before"], "base_mean": r["base_mean"],
                        "set": "control"})
        return out

    def claim(self, rule_id):
        """queued -> testing, only if nobody changed the rule meanwhile. Returns the row or None."""
        with Tx(self.db):
            self.db.execute("UPDATE rule SET status='testing' WHERE id=? AND status IN ('queued','testing')", (rule_id,))
            row = self.db.execute("SELECT status FROM rule WHERE id=?", (rule_id,)).fetchone()
        return row is not None and row["status"] == "testing"

    def finish(self, rule_id, decision, gate, run_fields, actor="code"):
        """One transaction: the gate_run row, the rule's status and notes (only if it is still `testing`: a withdraw or an enable by the
        person while the night ran wins, and the verdict goes to the journal only), the journal line. Returns (gate_run_id, applied)."""
        with Tx(self.db):
            run_id = self._gate_run(**run_fields)
            gate = dict(gate, run=run_id)
            row = self.db.execute("SELECT status, notes FROM rule WHERE id=?", (rule_id,)).fetchone()
            notes = json.loads(row["notes"] or "{}")
            key = notes.get("key")
            applied = row["status"] == "testing"
            if applied:
                notes["gate"] = gate
                notes.pop("gate_skipped", None)
                if decision == "rejected":
                    notes["gate_rejected"] = True
                self.db.execute("UPDATE rule SET status=?, notes=? WHERE id=? AND status='testing'", (decision, json.dumps(notes, ensure_ascii=False), rule_id))
                self.db.execute("INSERT INTO journal(ts, actor, event, ref) VALUES (?,?,?,?)", (now_iso(), actor, f"rule_{decision}", f"{key} gate run {run_id}"))
            else:
                self.db.execute("INSERT INTO journal(ts, actor, event, ref) VALUES (?,?,?,?)",
                                (now_iso(), actor, "rule_gate_ignored", f"{key} gate run {run_id} {decision}: the rule was {row['status']} meanwhile, not changed"))
            self.db.execute("UPDATE gate_run SET reason=? WHERE id=?", (f"rule {key}: {run_fields.get('reason')}", run_id))
        return run_id, applied

    def note_ablation(self, rule_id, gate, run_fields):
        with Tx(self.db):
            run_id = self._gate_run(**run_fields)
            row = self.db.execute("SELECT notes FROM rule WHERE id=?", (rule_id,)).fetchone()
            notes = json.loads(row["notes"] or "{}")
            notes["ablation"] = dict(gate, run=run_id)
            self.db.execute("UPDATE rule SET notes=? WHERE id=?", (json.dumps(notes, ensure_ascii=False), rule_id))
            self.db.execute("INSERT INTO journal(ts, actor, event, ref) VALUES (?,?,?,?)", (now_iso(), "code", "rule_ablation", f"{notes.get('key')} {gate.get('decision')} gate run {run_id}"))
            self.db.execute("UPDATE gate_run SET reason=? WHERE id=?", (f"ablation of {notes.get('key')}: {gate.get('decision')}", run_id))
        return run_id

    def skip(self, rule_id, reason):
        with Tx(self.db):
            row = self.db.execute("SELECT notes FROM rule WHERE id=?", (rule_id,)).fetchone()
            notes = json.loads(row["notes"] or "{}")
            if (notes.get("gate_skipped") or {}).get("reason") == reason:
                return                     # the same reason as last night: nothing new to write
            notes["gate_skipped"] = {"reason": reason, "at": now_iso()}
            self.db.execute("UPDATE rule SET notes=? WHERE id=?", (json.dumps(notes, ensure_ascii=False), rule_id))
            self.db.execute("INSERT INTO journal(ts, actor, event, ref) VALUES (?,?,?,?)", (now_iso(), "code", "rule_gate_skipped", f"{notes.get('key')}: {reason}"))

    def _gate_run(self, **fields):
        cols = ["started", "finished", "model", "set_id", "split", "repeats", "n_tasks", "base_mean", "cand_mean", "delta", "ci_low",
                "resolved_delta_low", "time_ratio", "tokens_ratio", "decision", "reason", "raw_path"]
        cur = self.db.execute(f"INSERT INTO gate_run({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(fields.get(c) for c in cols))
        return cur.lastrowid

    def close(self):
        self.db.close()


class Tx:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.db.execute("COMMIT" if exc_type is None else "ROLLBACK")
        return False


def render_rules(rows):
    """The personal layer's text for a list of rule rows, the way mcp/rules.py renders it (same header, one line per rule)."""
    rows = sorted(rows, key=lambda r: (r["enabled_at"] or "", r["id"]))
    if not rows:
        return ""
    lines = ["Personal rules of this model, confirmed by the person and passed the check (they add to the base above, never replace it):"]
    lines += [f"- {' '.join(str(r['text']).split())}" for r in rows]
    return "\n".join(lines) + "\n"


def layered(base_text, layer_text):
    """What the model gets: the base harness, then the personal layer as a second block (two system messages in the product)."""
    base_text = base_text.rstrip("\n")
    return (base_text + "\n\n" if base_text else "") + layer_text if layer_text else (base_text + "\n" if base_text else "")


def delivered(rec):
    d = rec.get("harness_delivery") or {}
    return bool(d.get("delivered")) or (not d.get("checked") and rec.get("harness_delivered") is True)


class LifelongGate(pilot.Pilot):
    """One night, one candidate (or one ablation), saved step by step like every rsi night (a rerun resumes).

    Steps: candidate -> tasks and sets (control tasks with room; safety = the tasks solved before) -> base runs on the current layer ->
    candidate runs -> decide2 (k = 1), safety_ok, cost (decisive) -> gate_run row, rule status, journal, EXPERIENCE.md, notice.
    Prior nights' base runs of the same layer fingerprint are reused after rsi2.drift_ok on DRIFT_TASKS fresh runs."""

    def __init__(self, name, model, runner, *, store, skills_dir=DEFAULT_SKILLS, candidate=None, ablate=None, prior_runs=(), max_growth=None, **kw):
        kw.setdefault("thresholds", accept.STUB)
        kw.setdefault("allow_stub", kw["thresholds"] is accept.STUB)
        super().__init__(name, model, {"train": [], "dev": [], "test": []}, runner, **kw)
        self.store, self.skills_dir = store, Path(skills_dir)
        self.candidate_key, self.ablate_key, self.prior_runs, self.max_growth = candidate, ablate, tuple(prior_runs), max_growth

    # ---- what is measured -----------------------------------------------------------------------------------------------------
    def _base_text(self):
        return (self.h_dir / "harness.md").read_text(encoding="utf-8") if (self.h_dir / "harness.md").exists() else ""

    def _root(self, label, layer_text):
        """A harness root <root>/<model>/harness.md with the base harness and the given layer text; the rest of the base folder as it is."""
        root = self.dir / "trials" / label
        if (root / self.model).exists():
            pilot.shutil.rmtree(root / self.model)
        pilot.copy_tree(self.h_dir, root / self.model)
        (root / self.model / "harness.md").write_text(layered(self._base_text(), layer_text), encoding="utf-8")
        return root

    def _pick(self):
        enabled = self.store.rules(self.model, ("enabled",))
        if self.ablate_key:
            victim = next((r for r in enabled if r["notes"]["key"] == self.ablate_key), None)
            if victim is None:
                raise pilot.Stop(f"ablation: no enabled rule '{self.ablate_key}' for {self.model}")
            return {"mode": "ablation", "rule_id": victim["id"], "key": victim["notes"]["key"], "text": victim["text"],
                    "current": [r["notes"]["key"] for r in enabled],
                    "base_layer": render_rules(enabled), "cand_layer": render_rules([r for r in enabled if r["id"] != victim["id"]])}
        queued = self.store.rules(self.model, ("queued", "testing"))
        if self.candidate_key:
            queued = [r for r in queued if r["notes"]["key"] == self.candidate_key]
        if not queued:
            raise pilot.Stop(f"no queued rule for {self.model}")
        for cand in queued:                             # the oldest first: one candidate per night (§4.6)
            replaced = set(cand["notes"].get("merged_from") or [])
            kept = [r for r in enabled if r["notes"]["key"] not in replaced]
            with_cand = kept + [dict(cand, enabled_at=now_iso())]
            # a candidate the person could not enable afterwards (§4.7 limits) is not worth a night: skipped with the reason, next one
            if len(kept) >= MAX_ENABLED:
                self.store.skip(cand["id"], f"the layer holds {len(kept)} rules already (limit {MAX_ENABLED}); merge first")
                continue
            tokens = est_tokens(render_rules(with_cand))
            if tokens > TOKEN_BUDGET:
                self.store.skip(cand["id"], f"the layer would be ~{tokens} tokens (budget {TOKEN_BUDGET}); merge first")
                continue
            if not self.store.claim(cand["id"]):
                continue                                # changed by the person meanwhile
            return {"mode": "candidate", "rule_id": cand["id"], "key": cand["notes"]["key"], "text": cand["text"], "merged_from": sorted(replaced),
                    "current": [r["notes"]["key"] for r in enabled], "base_layer": render_rules(enabled), "cand_layer": render_rules(with_cand)}
        raise pilot.Stop(f"no measurable queued rule for {self.model} (the queued ones do not fit the layer's limits; see their notes)")

    def _sets(self, tasks):
        """Growth: control tasks with room (base_mean from the store when known, else measured by a one-repeat screen); safety: the tasks
        solved before (solved_before = 1). A task may be in both only as safety."""
        known = {t["id"]: t["base_mean"] for t in tasks if t.get("base_mean") is not None}
        unknown = [t for t in tasks if t["id"] not in known]
        screen = {}
        if unknown:
            fp = executor.fingerprint(self.h_dir)
            recs = self._records("screen", self.h_root, fp, unknown, 1)
            screen = {r["task"]: r["score"] for r in recs}
        means = {**known, **screen}
        safety = [t["id"] for t in tasks if t.get("solved_before") == 1]
        growth = sorted((t["id"] for t in tasks if t["id"] not in set(safety) and means.get(t["id"], 1.0) <= GROWTH_MAX_MEAN), key=lambda i: (means.get(i, 1.0), i))
        if self.max_growth:
            growth = growth[: self.max_growth]
        return {"growth": growth, "safety": safety, "means": {k: round(v, 3) for k, v in means.items()}}

    def _records(self, label, root, fp, tasks, reps):
        out = self.dir / "runs" / label / fp
        recs = [self._run_step(f"{label}.{fp}.{t['id']}.{rep}", t, root, f"{label}-{fp}", rep, out) for t in tasks for rep in range(1, reps + 1)]
        for r in recs:
            if not delivered(r):           # RSI2D: a run the harness never reached is not a measurement; the night stops
                raise pilot.Stop(f"harness not delivered in {r['task']} {label} {r.get('rep')}: the run is not counted")
        return recs

    def _reusable_base(self, fp, tasks):
        """Base records of an earlier night under the same layer fingerprint, trusted after a drift check on a few fresh runs."""
        reused = []
        for name in self.prior_runs:
            st = driver.load_state(name)
            for key, step in st.get("steps", {}).items():
                if key.startswith(f"base.{fp}.") and isinstance(step.get("value"), dict) and step["value"].get("task") in {t["id"] for t in tasks}:
                    reused.append(step["value"])
        return reused

    # ---- the night ------------------------------------------------------------------------------------------------------------
    def run_cycle(self):
        self.meta.setdefault("started", time.time())
        self.meta.update({"scheme": "lifelong", "model": self.model, "fake": self.fake, "repeats": self.repeats, "status": "running",
                          "db": str(self.store.path), "skills_dir": str(self.skills_dir), "prior_runs": list(self.prior_runs)})
        self._save()
        try:
            status = self._night()
        except pilot.Stop as stop:
            status = f"stopped: {stop}"
            self._release_testing()
        self.meta.update(status=status, ended=time.time())
        self._save()
        return status

    def _release_testing(self):
        """A night that stopped leaves no rule in `testing`: it goes back to the queue."""
        c = self.state["steps"].get("candidate", {}).get("value")
        if c and c.get("mode") == "candidate":
            self.store.db.execute("UPDATE rule SET status='queued' WHERE id=? AND status='testing'", (c["rule_id"],))

    def _night(self):
        cand = self.step("candidate", self._pick)
        if cand["mode"] == "candidate" and not self.store.claim(cand["rule_id"]):
            raise pilot.Stop(f"rule '{cand['key']}' is no longer queued or testing (the person decided meanwhile): nothing to measure")
        # a resumed night measures against the layer as it was; if the person changed the layer meanwhile the old runs are no measurement
        if render_rules(self.store.rules(self.model, ("enabled",))) != cand["base_layer"]:
            raise pilot.Stop("the personal layer changed since the night started (the person enabled or retired a rule): start a new night")
        tasks = self.store.control_tasks("work")
        if not tasks:
            raise pilot.Stop("no control tasks (split work) in the store: LL3a's set is not imported yet")
        by_id = {t["id"]: t for t in tasks}
        sets = self.step("sets", lambda: self._sets(tasks))
        growth = [by_id[i] for i in sets["growth"]]
        safety = [by_id[i] for i in sets["safety"]]
        if len(growth) < MIN_GROWTH:
            raise pilot.Stop(f"only {len(growth)} control tasks with room (need {MIN_GROWTH}): nothing to measure tonight")
        if len({t["family"] for t in growth}) < MIN_FAMILIES:
            raise pilot.Stop(f"the tasks with room come from {len({t['family'] for t in growth})} families (need {MIN_FAMILIES}): the family bound would be no measurement")
        base_root, cand_root = self._root("base", cand["base_layer"]), self._root("cand", cand["cand_layer"])
        fb, fc = executor.fingerprint(base_root / self.model), executor.fingerprint(cand_root / self.model)
        self.meta.update(base_fingerprint=fb, cand_fingerprint=fc, k=1)
        # the base: reused from an earlier night when the drift check passes, else run now
        reused = self._reusable_base(fb, growth + safety)
        if reused:
            fresh = self._records("base", base_root, fb, growth[: rsi2.DRIFT_TASKS], 1)
            drift = self.step("drift", lambda: rsi2.drift_ok(fresh, reused))
            if drift["ok"]:
                base_g = self._fill("base", base_root, fb, growth, [r for r in reused if r["task"] in sets["growth"]])
                base_s = self._fill("base", base_root, fb, safety, [r for r in reused if r["task"] in sets["safety"]])
            else:
                base_g, base_s = self._records("base", base_root, fb, growth, self.repeats), self._records("base", base_root, fb, safety, self.repeats)
        else:
            base_g, base_s = self._records("base", base_root, fb, growth, self.repeats), self._records("base", base_root, fb, safety, self.repeats)
        cand_g = self._records("cand", cand_root, fc, growth, self.repeats)
        cand_s = self._records("cand", cand_root, fc, safety, self.repeats) if safety else []
        if safety and len(base_s) < len(safety) * self.repeats:
            raise pilot.Stop("the safety set has no complete base: refusing to decide without it")
        verdict = self.step("verdict", lambda: self._verdict(cand, base_g, cand_g, base_s, cand_s, sets))
        means = {"base": round(statistics.mean(r["score"] for r in base_g), 4), "cand": round(statistics.mean(r["score"] for r in cand_g), 4)}
        self.step("record", lambda: self._record(cand, verdict, sets, means))
        return f"done: {cand['mode']} '{cand['key']}' {verdict['decision']}"

    def _fill(self, label, root, fp, tasks, have_recs):
        """Reused records plus fresh runs for every (task, rep) they do not cover: the set is always complete."""
        have = {(r["task"], r.get("rep", 1)) for r in have_recs}
        missing = [t for t in tasks if not all((t["id"], rep) in have for rep in range(1, self.repeats + 1))]
        fresh = self._records(label, root, fp, missing, self.repeats) if missing else []
        return have_recs + [r for r in fresh if (r["task"], r["rep"]) not in have]

    def _verdict(self, cand, base_g, cand_g, base_s, cand_s, sets):
        if cand["mode"] == "ablation":
            # the layer without the rule, against the layer with it: "no benefit" when the lower bound of (without - with) is above -0.03
            v = rsi2.decide2(cand_g, base_g, k=1, thresholds=self.thresholds, allow_stub=self.allow_stub, cost_decides=False)
            no_benefit = v["score"]["lower"] > -0.03 and v["resolved"]["lower"] > -0.03
            return {"mode": "ablation", "decision": "no_benefit" if no_benefit else "has_benefit", "score": v["score"], "resolved": v["resolved"],
                    "time_ratio": v["time_ratio"], "tokens_ratio": v["tokens_ratio"], "pairs": v["pairs"], "stub": v["stub"]}
        v = rsi2.decide2(cand_g, base_g, k=1, thresholds=self.thresholds, allow_stub=self.allow_stub, cost_decides=True)
        s = rsi2.safety_ok(cand_s, base_s) if base_s and cand_s else {"ok": True, "pairs": 0, "mean": 0.0, "broke": []}
        ok = bool(v["accept"] and s["ok"])
        reason = "passed the pre-registered rule" if ok else ("; ".join(x for x, bad in (("score bound not above zero", not v["score"]["ok"]),
                                                                                       ("resolved bound below -0.03", not v["resolved"]["ok"]),
                                                                                       ("cost over the limit", not v["cost_ok"]),
                                                                                       ("a family regressed", bool(v["family_regressions"])),
                                                                                       ("the safety set dropped", not s["ok"])) if bad) or v.get("reason", "refused"))
        return {"mode": "candidate", "decision": "passed" if ok else "rejected", "reason": reason, "score": v["score"], "resolved": v["resolved"],
                "time_ratio": v["time_ratio"], "tokens_ratio": v["tokens_ratio"], "cost_ok": v["cost_ok"], "family_regressions": v["family_regressions"],
                "safety": s, "pairs": v["pairs"], "families": v["families"], "stub": v["stub"], "rule": v["rule"]}

    def _record(self, cand, verdict, sets, means):
        run_fields = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.meta["started"])), finished=now_iso(), model=self.model,
                          set_id="control:work", split="work", repeats=self.repeats, n_tasks=len(sets["growth"]), base_mean=means["base"], cand_mean=means["cand"],
                          delta=verdict["score"]["mean"], ci_low=verdict["score"]["lower"], resolved_delta_low=verdict["resolved"]["lower"],
                          time_ratio=verdict["time_ratio"], tokens_ratio=verdict["tokens_ratio"], decision=verdict["decision"],
                          reason=verdict.get("reason") or verdict["decision"], raw_path=str(driver.state_path(self.name)))
        gate = {"decision": verdict["decision"], "delta": verdict["score"]["mean"], "ci_low": verdict["score"]["lower"], "time_ratio": verdict["time_ratio"],
                "tokens_ratio": verdict["tokens_ratio"], "n_tasks": len(sets["growth"]), "repeats": self.repeats, "stub": verdict.get("stub", False), "at": now_iso()}
        applied = True
        if cand["mode"] == "ablation":
            run_id = self.store.note_ablation(cand["rule_id"], gate, run_fields)
            line = f"ablation of '{cand['key']}' (gate run {run_id}): {verdict['decision']} (score without it {verdict['score']['mean']}, lower {verdict['score']['lower']}); nothing retired by code"
        else:
            run_id, applied = self.store.finish(cand["rule_id"], verdict["decision"], gate, run_fields)
            line = (f"gate run {run_id}: rule '{cand['key']}' {verdict['decision']} ({verdict.get('reason')}); score {verdict['score']['mean']} "
                    f"(lower {verdict['score']['lower']}), time x{verdict['time_ratio']}, tokens x{verdict['tokens_ratio']}, {len(sets['growth'])} tasks x {self.repeats}")
            if not applied:
                line += "; the person changed the rule meanwhile, the status was left alone"
        self._experience(line)
        notice = None
        if verdict["decision"] == "passed" and applied:
            notice = (f"Правило «{cand['key']}» прошло ночную проверку (прогон {run_id}). Включить? В терминале: "
                      f"python3 <shuracode>/mcp/rules.py enable {cand['key']} --model {self.model}")
            if not self.fake:
                self._notify(notice)
        return {"gate_run_id": run_id, "notice": notice, "experience": line, "applied": applied}

    def _experience(self, line):
        folder = self.skills_dir / self.model
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "EXPERIENCE.md", "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(f"# Personal layer of {self.model}: what happened and why\n\n")
            f.write(f"- {now_iso()[:19]} {line}\n")
        try:
            os.chmod(folder / "EXPERIENCE.md", 0o600)
        except OSError:
            pass

    def _notify(self, text):
        """DESIGN §11 p. 1: the person hears that a version passed and decides. notify-send when there is a desktop; never fatal."""
        import shutil
        import subprocess
        if shutil.which("notify-send"):
            try:
                subprocess.run(["notify-send", "Shura", text], timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                pass


class FakeRuleRunner:
    """The plumbing check (--fake): a task scores 1.0 when the candidate's marker text is in the harness the run was given, 0.75
    otherwise; every record says the harness was delivered. Solved-before tasks score 1.0 with or without the marker."""

    def __init__(self, marker="FAKE-RULE", help_score=1.0, base_score=0.75):
        self.marker, self.help_score, self.base_score, self.calls = marker, help_score, base_score, []

    def run(self, task, root, model, arm, rep, out):
        text = (Path(root) / model / "harness.md").read_text(encoding="utf-8")
        good = self.marker in text
        self.calls.append((task["id"], arm, rep, good))
        Path(out).mkdir(parents=True, exist_ok=True)
        score = 1.0 if task.get("solved_before") == 1 else (self.help_score if good else self.base_score)
        return {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "model": model, "harness": executor.fingerprint(Path(root) / model),
                "resolved": score >= 1, "score": score, "seconds": 10.0, "tokens": 100, "requests": 2, "timed_out": False, "test_tail": "",
                "harness_delivery": {"checked": True, "delivered": True}}


def run_cli(args):
    """`rsi.driver --run NAME lifelong [--model M] [--fake] [--db PATH] [--skills-dir PATH] [--candidate KEY | --ablate KEY] [--prior-runs a,b]`."""
    if args.candidate and args.ablate:
        raise SystemExit("--candidate and --ablate exclude each other")
    store = RuleStore(args.db)
    try:
        if args.fake:
            runner = FakeRuleRunner()
            thresholds, allow_stub = accept.STUB, True
        else:
            raise SystemExit("the real stand for the lifelong gate is wired in LL5 (gateway stand + control tasks on GPU); use --fake for the plumbing check")
        gate = LifelongGate(args.run, args.model, runner, store=store, skills_dir=args.skills_dir, candidate=args.candidate, ablate=args.ablate,
                            prior_runs=[x for x in (args.prior_runs or "").split(",") if x], thresholds=thresholds, allow_stub=allow_stub,
                            repeats=args.repeats, fake=args.fake, timebox_min=args.timebox_min, crash_after=args.crash_after)
        status = gate.run_cycle()
        print(f"lifelong {args.run}: {status}")
        rec = driver.load_state(args.run)["steps"].get("record", {}).get("value")
        if rec and rec.get("notice"):
            print(rec["notice"])
        return 0
    finally:
        store.close()
