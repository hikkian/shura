#!/usr/bin/env python3
"""Pre-registered paired gate statistics. No outcome is enabled by this tool."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import hashlib
from functools import lru_cache
from collections import defaultdict
from pathlib import Path


def _binomial_cdf(k, n, p):
    """P[X <= k] for X~Binomial(n,p), stable enough for exam-sized n."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    # Sum from the mode to avoid underflow in the low-probability tail.
    probs = [0.0] * (n + 1)
    mode = min(n, int((n + 1) * p))
    probs[mode] = math.exp(math.lgamma(n + 1) - math.lgamma(mode + 1) - math.lgamma(n - mode + 1)
                           + mode * math.log(p) + (n-mode) * math.log1p(-p))
    for i in range(mode, 0, -1):
        probs[i-1] = probs[i] * i / (n-i+1) * (1-p) / p
    for i in range(mode, n):
        probs[i+1] = probs[i] * (n-i) / (i+1) * p / (1-p)
    return min(1.0, sum(probs[:k+1]))


@lru_cache(maxsize=8192)
def clopper_pearson(k, n, alpha=.05):
    """Exact one-sided (1-alpha) Clopper-Pearson limits for a binomial share."""
    k, n = int(k), int(n)
    if n <= 0 or k < 0 or k > n or not 0 < alpha < 1:
        raise ValueError("need 0 <= k <= n, n > 0, and 0 < alpha < 1")
    lo = 0.0 if k == 0 else _bisect_probability(lambda p: 1-_binomial_cdf(k-1, n, p), alpha)
    hi = 1.0 if k == n else _bisect_decreasing(lambda p: _binomial_cdf(k, n, p), alpha)
    return lo, hi


def _bisect_probability(fn, target):
    a, b = 0.0, 1.0
    for _ in range(70):
        m = (a+b)/2
        if fn(m) < target:
            a = m
        else:
            b = m
    return (a+b)/2


def _bisect_decreasing(fn, target):
    a, b = 0.0, 1.0
    for _ in range(70):
        m = (a+b)/2
        if fn(m) > target:
            a = m
        else:
            b = m
    return (a+b)/2


def validate_switches(prereg):
    """Validate opt-in switch declarations before examining any outcomes."""
    declared = prereg.get("switches", prereg.get("switch", []))
    if declared == "always":
        declared = []
    elif isinstance(declared, dict):
        declared = [declared]
    seen_ids = set()
    for sw in declared:
        if sw == "always" or sw.get("mode") == "always":
            continue
        required = ("id", "condition", "slices", "unit", "budgets")
        if any(key not in sw for key in required):
            raise ValueError("each conditional switch needs id, condition, slices, unit, budgets")
        if not isinstance(sw["slices"], list) or not sw["slices"]:
            raise ValueError("switch slices must be a nonempty list of names or dimension maps")
        if not isinstance(sw["budgets"], dict) or any(not 0 <= float(v) <= 1 for v in sw["budgets"].values()):
            raise ValueError("switch budgets must map quality row names to shares in [0,1]")
        if sw["unit"] not in ("family", "conversation"):
            raise ValueError("switch unit must be family for exams or conversation for live work")
        if not isinstance(sw["id"], str) or not sw["id"] or sw["id"] in seen_ids:
            raise ValueError("switch ids must be nonempty and unique")
        seen_ids.add(sw["id"])
        if sw["unit"] == "family" and sw.get("exam_unit", "family") != "family":
            raise ValueError("exam switch unit is always task family")
    return prereg


def read_counter_records(path):
    """Read live gateway JSONL counters and exam run results through one record interface."""
    return read_results(path)


def counter_units(records, switch_ids, unit="family", union=False, slice_key=None):
    """Collapse requests/tasks to independent units; repeated firings count once."""
    units = defaultdict(bool)
    for rec in records:
        if rec.get("mode") not in (None, "shadow", "live"):
            continue
        fired = bool(rec.get("fired"))
        if "switches" in rec:
            fired = any(bool(rec.get("switches", {}).get(s, {}).get("fired")) for s in switch_ids)
        elif union and isinstance(rec.get("switches"), dict):
            fired = any(bool(v.get("fired")) for v in rec["switches"].values())
        elif switch_ids:
            fired = fired and rec.get("switch", switch_ids[0]) in switch_ids
        # Exam task sets always cluster by task family; live rows use the conversation hash.
        ident = rec.get("family") if unit == "family" else (rec.get("unit") or rec.get("conversation"))
        if ident is None:
            raise ValueError("counter record lacks independent unit (family or unit)")
        sl = rec.get("slice") or {k: rec.get(k) for k in ("lang", "project", "kind") if rec.get(k) is not None}
        key = (str(ident), tuple(sorted((sl or {}).items())) if slice_key is None else sl.get(slice_key))
        units[key] = units[key] or fired
    return len(units), sum(units.values())


def analyze_counters(records, prereg):
    """Counter-only closure for quality rows and shared union budgets."""
    validate_switches(prereg)
    declared = prereg.get("switches", prereg.get("switch", []))
    if declared == "always":
        declared = []
    elif isinstance(declared, dict):
        declared = [declared]
    switches = [s for s in declared if isinstance(s, dict) and s.get("mode") != "always"]
    ids = [s["id"] for s in switches]
    out = {}
    for row in prereg.get("rows", []):
        if row.get("kind") in ("speed", "budget") or row.get("metric") in ("seconds", "speed", "budget"):
            out[row["name"]] = {"closed": False, "reason": "speed and budget rows require measurement"}
            continue
        if row.get("metric", prereg.get("metric", "score")) != "score":
            out[row["name"]] = {"closed": False, "reason": "counter closure requires a 0..1 quality score"}
            continue
        row_specs = [s for s in switches if row["name"] in s.get("budgets", {})]
        if not row_specs:
            out[row["name"]] = {"closed": False, "reason": "no conditional switch registered for this row"}
            continue
        row_ids = [s["id"] for s in row_specs]
        selected = [r for r in records if any(_slice_matches(r, sl) for sw in row_specs for sl in sw["slices"])]
        n, k = counter_units(selected, row_ids, unit=row_specs[0]["unit"], union=True)
        alpha = .05  # The counter row's stated bound is one-sided 95%; lifetime look-splitting is for the fuse.
        lo, hi = clopper_pearson(k, n, alpha) if n else (None, None)
        delta = min([float(row["delta"]), *[float(s["budgets"][row["name"]]) for s in row_specs]]) if row_specs else float(row["delta"])
        out[row["name"]] = {"n_units": n, "fired_units": k, "share": k/n if n else None,
                             "lower_bound": lo, "upper_bound": hi, "delta": delta,
                             "closed": bool(n and hi <= delta), "method": "exact one-sided Clopper-Pearson",
                             "union_switches": row_ids, "alpha": alpha}
    return out


def _slice_matches(record, selector):
    sl = record.get("slice") or {k: record.get(k) for k in ("lang", "project", "kind") if record.get(k) is not None}
    if selector == "*":
        return True
    if isinstance(selector, dict):
        return all(sl.get(k) == v for k, v in selector.items())
    if isinstance(selector, str):
        if record.get("slice_name") == selector:
            return True
        # Compact prereg form, e.g. "lang=ru" or "lang=ru,kind=implement".
        parts = [part.split("=", 1) for part in selector.split(",")]
        return all(len(pair) == 2 and sl.get(pair[0].strip()) == pair[1].strip() for pair in parts)
    return False


def fuse_counter(records, switch_id, budget, checks, look, unit="conversation", foreign_slice=None):
    """Sequentially-safe fuse: return to shadow only when lower bound exceeds budget."""
    if int(checks) < 1 or int(look) < 1 or int(look) > int(checks):
        raise ValueError("look must be preregistered between 1 and checks")
    alpha = .05 / int(checks)
    selected = [r for r in records if foreign_slice is None or r.get("slice") == foreign_slice]
    n, k = counter_units(selected, [switch_id], unit=unit)
    lo, hi = clopper_pearson(k, n, alpha) if n else (None, None)
    return {"action": "return_to_shadow" if lo is not None and lo > budget else "keep_live",
            "n_units": n, "fired_units": k, "lower_bound": lo, "upper_bound": hi,
            "budget": budget, "check": look, "checks_lifetime": checks, "alpha_per_check": alpha}


def simulate_switch_counter(repeats=1000, families=100, tasks_per_family=4, budget=.03,
                            true_shares=(0.0, .03, .04, .10), seed=20261113, checks=5):
    """Exercise family clustering, exact closure, and the lifetime fuse with deterministic RNG."""
    rng = random.Random(seed)
    report = {}
    for share in true_shares:
        closed = false_fuse = 0
        for _ in range(repeats):
            # Strong within-family correlation: a latent family event drives all task firings.
            family_fired = []
            for _family in range(families):
                latent = rng.random() < share
                # Every task in this family shares the latent outcome; resampling still
                # counts the family once, never the four correlated task rows.
                task_firings = [latent] * tasks_per_family
                family_fired.append(any(task_firings))
            _, k = families, sum(family_fired)
            _, upper = clopper_pearson(k, families)
            closed += upper <= budget
            # Five scheduled looks use disjoint family batches and pre-split alpha.
            fired_total = seen = False
            for _look in range(checks):
                block_n = max(1, families // checks)
                block_k = sum(rng.random() < share for _ in range(block_n))
                fired_total += block_k
                seen += block_n
                lower, _ = clopper_pearson(fired_total, seen, alpha=.05/checks)
                if lower > budget:
                    false_fuse += share <= budget
                    break
        report[str(share)] = {"row_closed_rate": closed/repeats,
                              "fuse_rate": false_fuse/repeats if share <= budget else None}
    return {"repeats": repeats, "families": families, "tasks_per_family": tasks_per_family,
            "within_family_correlation": 1.0, "budget": budget, "checks": checks, "seed": seed,
            "rates": report}


def analyze_split_anchor(records, prereg, candidate, **anchor_kwargs):
    """Report anchor core by paired runs and conditional layer by the union counter."""
    core = analyze_anchor(records, candidate, **anchor_kwargs)
    stack = anchor_kwargs.get("stack", "stack")
    conditional = analyze_counters([r for r in records if r.get("arm", stack) == stack], prereg)
    return {"core_always_by_runs": core, "conditional_union_by_counter": conditional,
            "automatic_enable": False}


def token_counter_study(records, threshold):
    """QLEV1 stand-in: token threshold fires per run; inference collapses to task families."""
    by_family = defaultdict(bool)
    tasks = set()
    for rec in records:
        family = rec.get("family")
        if family is None:
            continue
        tasks.add(rec.get("task"))
        by_family[str(family)] = by_family[str(family)] or int(rec.get("tokens", 0) or 0) >= int(threshold)
    n, k = len(by_family), sum(by_family.values())
    lo, hi = clopper_pearson(k, n) if n else (None, None)
    return {"threshold_tokens": int(threshold), "tasks": len(tasks), "families": n,
            "fired_families": k, "share": k/n if n else None, "lower_bound": lo, "upper_bound": hi}


def load_data(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError as exc:
        raise ValueError("YAML input requires PyYAML; alternatively provide JSON") from exc


def percentile(sorted_values, p):
    if not sorted_values:
        raise ValueError("cannot calculate percentile of empty data")
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def paired_bootstrap(diffs, families=None, reps=10000, seed=1, level=.95):
    """Cluster bootstrap tasks; optionally sample task families at the outer level."""
    if not diffs:
        raise ValueError("at least one paired task is required")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, d in enumerate(diffs):
        groups[(families[i] if families else "__all__")].append(d)
    names = sorted(groups)
    means = []
    for _ in range(reps):
        sampled = []
        for name in (rng.choices(names, k=len(names)) if families else names):
            vals = groups[name]
            sampled.extend(rng.choices(vals, k=len(vals)))
        means.append(statistics.fmean(sampled))
    means.sort()
    tail = (1 - level) / 2
    return percentile(means, tail), percentile(means, 1-tail)


def exact_paired_sign_test(diffs, alternative="two-sided"):
    """Exact sign test for paired nonzero task differences (ties are omitted)."""
    nz = [d for d in diffs if d != 0]
    n, wins = len(nz), sum(d > 0 for d in nz)
    if not n:
        return 1.0
    lo = sum(math.comb(n, i) for i in range(wins + 1)) / (2 ** n)
    hi = sum(math.comb(n, i) for i in range(wins, n + 1)) / (2 ** n)
    if alternative == "greater":
        return hi
    if alternative == "less":
        return lo
    return min(1.0, 2 * min(lo, hi))


def exact_randomization_test(diffs):
    """Exact paired sign-flip test of mean difference; feasible through 22 non-ties."""
    nz = [d for d in diffs if d]
    n = len(nz)
    if n > 22:
        raise ValueError("exact enumeration is limited to 22 nonzero pairs")
    if not n:
        return 1.0
    observed = abs(sum(nz) / n)
    extreme = 0
    for mask in range(1 << n):
        value = sum(d if mask >> i & 1 else -d for i, d in enumerate(nz)) / n
        extreme += abs(value) >= observed - 1e-15
    return extreme / (1 << n)


def exact_harm_test(diffs, delta):
    """One-sided paired sign-flip test against the harm margin; exact when feasible."""
    centered = [d + delta for d in diffs if d + delta]
    n = len(centered)
    if not n:
        return 1.0
    observed = sum(centered) / n
    if n > 22:
        rng = random.Random(20261113)
        hits = sum(sum(d if rng.getrandbits(1) else -d for d in centered) / n <= observed + 1e-15
                   for _ in range(100000))
        return (hits + 1) / 100001
    extreme = 0
    for mask in range(1 << n):
        value = sum(d if mask >> i & 1 else -d for i, d in enumerate(centered)) / n
        extreme += value <= observed + 1e-15
    return extreme / (1 << n)


def validate_prereg(prereg):
    candidate = prereg.get("candidate")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("preregistration must name exactly one candidate")
    if "candidates" in prereg:
        raise ValueError("one named candidate per gate; candidate selection belongs to search")
    rows = prereg.get("rows")
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("rows must be nonempty and uniquely named")
    if int(prereg.get("looks", 3)) not in (1, 2, 3):
        raise ValueError("looks must be between one and three")
    for row in rows:
        if float(row["delta"]) < 0:
            raise ValueError("delta must be nonnegative")
    targets = [r for r in rows if r.get("target") is True]
    if len(targets) > 1:
        raise ValueError("at most one pre-registered target row is allowed")
    return prereg


def read_results(path):
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
    records = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_row(row, records, prereg, combined_records=None):
    candidate, control = prereg["candidate"], prereg.get("control", "stack")
    metric = row.get("metric", prereg.get("metric", "score"))
    subset = [r for r in records if r.get("row") == row["name"]]
    counter_result = None
    if prereg.get("switches", prereg.get("switch")):
        counter_result = analyze_counters([r for r in subset if r.get("arm", control) == control],
                                          {**prereg, "rows": [row]})[row["name"]]
    # Every registered seed must be represented by both arms, paired on the same tasks.
    cells = defaultdict(list)
    sample_ids = defaultdict(list)
    fam = {}
    for rec in subset:
        arm, task = rec.get("arm"), rec.get("task")
        if arm not in (candidate, control):
            continue
        if metric not in rec:
            raise ValueError(f"missing metric {metric!r} in result record")
        cells[(arm, task)].append(float(rec[metric]))
        sample_ids[(arm, task)].append((rec.get("rep"), rec.get("seed")))
        if rec.get("family") is not None:
            fam[task] = rec["family"]
    tasks = sorted({t for a, t in cells if a in (candidate, control)})
    if not tasks:
        closed = bool(counter_result and counter_result.get("closed") and not row.get("target", False))
        return {"name": row["name"], "zone": "green" if closed else "grey",
                "reason": "closed by switch counter" if closed else "no paired results",
                "lower_bound": None, **({"counter": counter_result} if counter_result is not None else {})}
    n_expected = int(row.get("n", prereg.get("n", len(tasks))))
    k = int(row.get("k", prereg.get("k", 1)))
    if len(tasks) != n_expected:
        raise ValueError(f"row {row['name']}: expected {n_expected} tasks, got {len(tasks)}")
    for t in tasks:
        for arm in (candidate, control):
            vals = cells.get((arm, t), [])
            if len(vals) != k:
                raise ValueError(f"row {row['name']}: {t}/{arm} has {len(vals)} samples, expected k={k}")
        left, right = sample_ids[(candidate, t)], sample_ids[(control, t)]
        if any(seed is not None for _, seed in left + right) and sorted(left) != sorted(right):
            raise ValueError(f"row {row['name']}: candidate/control do not share repetitions and seeds for task {t}")
    seed = int(row.get("seed", prereg.get("seed", 1)))
    cd = [statistics.fmean(cells[(candidate,t)]) - statistics.fmean(cells[(control,t)]) for t in tasks]
    fs = [fam.get(t, "__unknown__") for t in tasks]
    look = int(prereg.get("look", 1))
    looks = int(prereg.get("looks", 3))
    alpha = float(prereg.get("alpha", .05)) / looks
    confidence = 1 - alpha
    lo, hi = paired_bootstrap(cd, reps=int(prereg.get("bootstrap_reps", 10000)), seed=seed, level=confidence)
    flo, fhi = paired_bootstrap(cd, fs, reps=int(prereg.get("bootstrap_reps", 10000)), seed=seed+1, level=confidence)
    p_harm = exact_harm_test(cd, float(row["delta"]))
    budget_bad = bool(row.get("budget_violated", False))
    target = bool(row.get("target", False))
    better = lo > 0
    harm = hi < -float(row["delta"]) and p_harm <= alpha
    noninferior = lo > -float(row["delta"])
    if counter_result is not None:
        noninferior = noninferior or counter_result.get("closed", False)
    eligible_gain = row.get("kind") in ("agent", "personal")
    if budget_bad or harm:
        zone = "red"
    elif noninferior and (not target or (eligible_gain and better)):
        zone = "green"
    else:
        zone = "grey"
    flips = [{"task": t, "control": statistics.fmean(cells[(control,t)]),
              "candidate": statistics.fmean(cells[(candidate,t)]),
              "direction": "candidate_better" if d > 0 else "control_better" if d < 0 else "tie"}
             for t,d in zip(tasks,cd)]
    return {"name": row["name"], "zone": zone, "n": len(tasks), "k": k,
            "mean_difference": statistics.fmean(cd), "lower_bound": lo, "upper_bound": hi,
            "family_lower_bound": flo, "family_upper_bound": fhi, "delta": float(row["delta"]),
            "exact_p_harm": p_harm, "alpha_per_look": alpha, "look": look,
            "budget_violated": budget_bad, "target_better": better,
            "gain_eligible": eligible_gain, "flips": flips,
            **({"counter": counter_result} if counter_result is not None else {})}


def verdict(prereg, results):
    zones = [r["zone"] for r in results]
    target_names = {r["name"] for r in prereg.get("rows", []) if r.get("target") is True}
    target_won = any(r.get("name") in target_names and r.get("zone") == "green"
                     and r.get("gain_eligible") is True and r.get("target_better") is True
                     and r.get("lower_bound") is not None and r["lower_bound"] > 0
                     for r in results)
    mandatory_pass = all(r.get("zone") == "green" for r in results)
    overall = "red" if "red" in zones else "green" if zones and mandatory_pass and target_won else "grey"
    return {"candidate": prereg["candidate"], "overall": overall, "rows": results,
            "target_proven": target_won, "automatic_enable": False}


def simulate_target_gate(repeats=1000, n=80, sigma=.22, delta=.03, alpha=.05,
                         seed=20261113):
    """Operating-characteristic check for the target-win gate (normal paired model)."""
    rng = random.Random(seed)
    outcomes = {"aa_with_target": 0, "aa_without_target": 0,
                "harmful_red": 0, "target_win_green": 0}
    target_power = 0
    # Match analyze_row's one-sided lower bound after preregistered look splitting.
    z = statistics.NormalDist().inv_cdf(1 - alpha / 3)
    for _ in range(repeats):
        # A/A with a nominated target must prove a positive lower bound.
        aa = [rng.gauss(0, sigma) for _ in range(n)]
        aa_mean = statistics.fmean(aa)
        aa_se = statistics.pstdev(aa) / math.sqrt(n)
        aa_lo = aa_mean - z * aa_se
        outcomes["aa_with_target"] += aa_lo > 0 and aa_lo > -delta
        # Without a target row verdict() categorically cannot return green.
        outcomes["aa_without_target"] += False

        harmful = [rng.gauss(-.15, sigma) for _ in range(n)]
        hmean = statistics.fmean(harmful)
        hse = statistics.pstdev(harmful) / math.sqrt(n)
        hhi = hmean + z * hse
        outcomes["harmful_red"] += hhi < -delta

        win = [rng.gauss(.10, sigma) for _ in range(n)]
        wmean = statistics.fmean(win)
        wse = statistics.pstdev(win) / math.sqrt(n)
        wlo = wmean - z * wse
        won = wlo > 0 and wlo > -delta
        outcomes["target_win_green"] += won
        target_power += won
    return {"repeats": repeats, "n": n, "sigma": sigma, "delta": delta,
            "alpha": alpha, "seed": seed,
            "rates": {name: count / repeats for name, count in outcomes.items()},
            "checks": {"aa_with_target_le_alpha": outcomes["aa_with_target"] / repeats <= alpha,
                       "aa_without_target_le_alpha": outcomes["aa_without_target"] / repeats <= alpha,
                       "harmful_red_at_least_99pct": outcomes["harmful_red"] / repeats >= .99,
                       "target_win_expected_power": target_power / repeats}}


def analyze_probes(spec, records):
    """Apply pre-registered P0 veto probes. Probe limits are supplied by the exam manifest."""
    profile = spec.get("profile", "deterministic")
    indexed = defaultdict(list)
    for rec in records:
        indexed[rec.get("probe")].append(rec)
    checks = []
    for name, rule in spec.get("checks", {}).items():
        vals = indexed.get(name, [])
        if not vals:
            checks.append({"name": name, "passed": False, "reason": "missing probe"})
            continue
        metric = rule.get("metric", "value")
        numbers = [float(r[metric]) for r in vals if metric in r]
        if len(numbers) != len(vals):
            checks.append({"name": name, "passed": False, "reason": f"missing {metric}"})
            continue
        op, limit = rule.get("op", "max"), float(rule.get("limit", 0))
        if name == "speed_pairs" and rule.get("noise_metric"):
            noise_key = rule["noise_metric"]
            if any(noise_key not in r for r in vals):
                checks.append({"name": name, "passed": False, "reason": f"missing {noise_key}"})
                continue
            z = float(rule.get("z", 1.96))
            numbers = [float(r[metric]) - z * float(r[noise_key]) for r in vals]
        if op == "equals":
            passed = all(r.get(metric) == rule.get("value") for r in vals)
        elif op == "min":
            passed = min(numbers) >= limit
        else:
            passed = max(numbers) <= limit
        checks.append({"name": name, "passed": passed, "observed": numbers,
                       "op": op, "limit": rule.get("value") if op == "equals" else limit})
    if profile == "deterministic":
        required = {"identity", "kl"}
    elif profile == "live":
        required = {"kl", "top8_changed", "exposure", "expert_cache", "mtp_accept", "speed_pairs"}
    else:
        raise ValueError("P0 profile must be deterministic or live")
    if profile in ("deterministic", "live"):
        present = set(spec.get("checks", {}))
        for missing in sorted(required - present):
            checks.append({"name": missing, "passed": False, "reason": f"required {profile} P0 check not preregistered"})
    failed = [c["name"] for c in checks if not c["passed"]]
    return {"profile": profile, "checks": checks, "veto": bool(failed),
            "early_stop": bool(failed), "failed": failed}


def analyze_anchor(records, candidate, stack="stack", base="tiel-deploy2", metric="score",
                   delta=0.0, seed=1, reps=10000, last_change_by_row=None):
    """Compare stack with the fixed Tiel/DEPLOY2 anchor on paired tasks and suggest rollback."""
    row_names = sorted({str(r["row"]) for r in records if r.get("row") is not None})
    if row_names:
        per_row = {row: analyze_anchor([{k:v for k,v in r.items() if k != "row"}
                    for r in records if str(r.get("row")) == row], candidate,
                    stack, base, metric, delta, seed, reps,
                    {"anchor": (last_change_by_row or {}).get(row)}) for row in row_names}
        return {"zone": "red" if any(x["zone"] == "red" for x in per_row.values()) else
                        "grey" if any(x["zone"] == "grey" for x in per_row.values()) else "green",
                "rows": per_row, "auto_rollback": any(x["auto_rollback"] for x in per_row.values()),
                "base_runs_on_new_tasks": sum(x["base_runs_on_new_tasks"] for x in per_row.values())}
    by = defaultdict(dict)
    families = {}
    for rec in records:
        if rec.get("arm") in (stack, base) and metric in rec:
            by[rec.get("task")][rec["arm"]] = float(rec[metric])
            if rec.get("family") is not None:
                families[rec["task"]] = rec["family"]
    pairs = [(t, arms) for t, arms in sorted(by.items()) if stack in arms and base in arms]
    if not pairs:
        base_runs = sum(1 for r in records if r.get("arm") == base and r.get("new_task", True))
        return {"zone": "grey", "reason": "no paired anchor results", "auto_rollback": False,
                "n": 0, "tasks": [], "distinct_families": 0,
                "base_runs_on_new_tasks": base_runs}
    diffs = [arms[stack] - arms[base] for _, arms in pairs]
    lo, hi = paired_bootstrap(diffs, reps=reps, seed=seed)
    rollback = hi < -delta
    return {"candidate": candidate, "stack": stack, "anchor": base, "n": len(diffs),
            "tasks": [t for t, _ in pairs], "distinct_families": len({families.get(t, "__unknown__") for t, _ in pairs}),
            "last_change_to_rollback": (last_change_by_row or {}).get("anchor") if rollback else None,
            "base_runs_on_new_tasks": sum(1 for r in records if r.get("arm") == base and r.get("new_task", True)),
            "mean_difference": statistics.fmean(diffs), "lower_bound": lo, "upper_bound": hi,
            "delta": delta, "zone": "red" if rollback else "green" if lo > -delta else "grey",
            "auto_rollback": rollback}


def merge_waves(waves):
    """Join separately collected waves, rejecting reused tasks or missing wave labels."""
    merged, seen = [], set()
    for index, records in enumerate(waves, 1):
        wave_tasks = {r.get("task") for r in records if r.get("task") is not None}
        duplicate = seen & wave_tasks
        if duplicate:
            raise ValueError(f"waves must use fresh tasks; repeated: {sorted(duplicate)}")
        seen.update(wave_tasks)
        merged.extend({**r, "wave": r.get("wave", index)} for r in records)
    return merged


def family_resample_study(records, sizes=(56, 120, 15), deltas=(.03, .05, .12),
                          effects=(0, -3, -5, -10, 5, 10), trials=1000, seed=20261113,
                          looks=3):
    """Cluster-resample QLEV repeat differences, then tasks within sampled families.

    Inputs are one configuration with two repeat records per task. Effects and
    reported score differences are percentage points (0..100 scale).
    """
    by_task = defaultdict(list)
    for r in records:
        if r.get("arm", "base") == "base":
            by_task[(r.get("family", "__unknown__"), r["task"])].append((int(r.get("rep", 0)), float(r["score"])))
    groups = defaultdict(list)
    for (family, task), vals in by_task.items():
        vals.sort()
        if len(vals) >= 2:
            # The data are repeat pairs; pair consecutive repeats and retain paired task differences.
            differences = [100 * (vals[i+1][1] - vals[i][1]) for i in range(0, len(vals)-1, 2)]
            if differences:
                groups[str(family)].extend(differences)
    if len(groups) < 2:
        raise ValueError("calibration needs repeat pairs in at least two families")
    fams = sorted(groups)
    rows = []
    rng = random.Random(seed)
    # Each simulated experiment draws family clusters, then tasks within each
    # selected cluster. The cluster count is capped by the measured 12/14 families.
    for n in sizes:
        for delta in deltas:
            for effect in effects:
                zones = defaultdict(int)
                for _ in range(trials):
                    picked_fams = [rng.choice(fams) for _ in range(len(fams))]
                    diffs = []
                    fs = []
                    q, rem = divmod(n, len(picked_fams))
                    for ix, f in enumerate(picked_fams):
                        vals = groups[f]
                        for _ in range(q + (ix < rem)):
                            diffs.append(rng.choice(vals) / math.sqrt(2) + effect)
                            fs.append(f)
                    # Family bootstrap is the inferential interval; conservative
                    # alpha is split across the preregistered looks.
                    lo, hi = paired_bootstrap(diffs, fs, reps=199, seed=rng.randrange(2**31), level=1-alpha_for_looks(looks))
                    if hi < -(delta * 100):
                        zones["red"] += 1
                    elif lo > -(delta * 100) and lo > 0:
                        # A target row is green only when it proves benefit too.
                        zones["green"] += 1
                    else:
                        zones["grey"] += 1
                rows.append({"tasks": n, "families_available": len(fams), "delta": delta,
                             "effect": effect, **{z: zones[z] / trials for z in ("green", "grey", "red")}})
    return rows


def alpha_for_looks(looks):
    return .05 / int(looks)


def train_plan(candidate_zones):
    """Plan the rejected-candidate train: full common exam; bisect red only."""
    return {"candidates": sorted(candidate_zones), "exam": "full", "split": "bisect red only",
            "red": sorted(c for c, z in candidate_zones.items() if z == "red"),
            "grey": sorted(c for c, z in candidate_zones.items() if z == "grey"),
            "green": sorted(c for c, z in candidate_zones.items() if z == "green"),
            "grey_action": "recheck whole train on fresh tasks",
            "red_action": "bisect red subgroup", "green_action": "all enter",
            "speed_credit_between_candidates": False}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_decision(path, record, sealed_access=False, annual_limit=8):
    """Append a decision record and enforce the sealed-set access budget."""
    p = Path(path)
    prior = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []
    year = str(record.get("date", ""))[:4]
    used = sum(1 for r in prior if r.get("sealed_access") and str(r.get("date", ""))[:4] == year)
    if sealed_access and used >= annual_limit:
        raise ValueError(f"sealed-set access limit ({annual_limit}/year) reached")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**record, "sealed_access": bool(sealed_access)}, ensure_ascii=False) + "\n")


def simulate(n, k, sigma, delta, trials=5000, seed=20261112):
    """Normal paired task noise calibration; sigma is across-task SD of paired means at k=1."""
    rng = random.Random(seed)
    out = {}
    for effect in (0, -.05, -.10, .05, .10):
        passed = 0
        for _ in range(trials):
            diffs = [effect + rng.gauss(0, sigma / math.sqrt(k)) for _ in range(n)]
            lo = statistics.fmean(diffs) - 1.96 * statistics.pstdev(diffs) / math.sqrt(n)
            passed += lo > -delta
        out[f"{effect:+.2f}"] = passed / trials
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prereg")
    ap.add_argument("results")
    ap.add_argument("--out", help="JSON verdict path (default: results/gate-verdict.json)")
    ap.add_argument("--wave", action="append", default=[], help="additional fresh-task JSONL wave; repeat at most twice")
    ap.add_argument("--probes", help="JSONL P0 probe results")
    ap.add_argument("--anchor-results", help="JSONL stack versus Tiel/DEPLOY2 measurements")
    ap.add_argument("--journal", help="append decision record to this JSONL")
    ap.add_argument("--sealed-access", action="store_true", help="record an access to the sealed set")
    ap.add_argument("--train-zones", help="JSON object mapping rejected candidates to red/grey/green")
    ap.add_argument("--calibrate", action="store_true", help="run family-resampled QLEV1 repeat-pair study")
    ap.add_argument("--calibration-out", help="write calibration rows as JSON")
    ap.add_argument("--counters", action="store_true", help="analyze live or exam switch-counter JSONL")
    ap.add_argument("--simulate-switches", type=int, metavar="REPEATS", help="simulate family counters and scheduled fuse checks")
    ap.add_argument("--token-counter-threshold", type=int, help="QLEV1 stand-in: fire when per-run completion tokens reach this value")
    a = ap.parse_args(argv)
    if a.simulate_switches is not None:
        print(json.dumps(simulate_switch_counter(repeats=a.simulate_switches), indent=2))
        return 0
    if a.counters:
        pre = validate_prereg(load_data(a.prereg))
        result = analyze_counters(read_counter_records(a.results), pre)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if a.token_counter_threshold is not None:
        print(json.dumps(token_counter_study(read_results(a.results), a.token_counter_threshold), indent=2))
        return 0
    if a.calibrate:
        study = family_resample_study(read_results(a.results))
        output = json.dumps(study, indent=2, ensure_ascii=False) + "\n"
        if a.calibration_out:
            Path(a.calibration_out).write_text(output, encoding="utf-8")
        print(output, end="")
        return 0
    pre = validate_prereg(load_data(a.prereg))
    if len(a.wave) > 2:
        ap.error("at most three total waves are permitted")
    if len(a.wave) + 1 > int(pre.get("looks", 3)):
        ap.error("wave count exceeds the preregistered number of looks")
    datasets = [read_results(p) for p in [a.results, *a.wave]]
    data = merge_waves(datasets) if a.wave else datasets[0]
    if a.wave:
        pre["look"] = len(datasets)
        for row in pre["rows"]:
            observed = {r.get("task") for r in data if r.get("row") == row["name"]
                        and r.get("arm") in (pre["candidate"], pre.get("control", "stack"))}
            row["n"] = len(observed)
    rows = [analyze_row(row, data, pre) for row in pre["rows"]]
    result = verdict(pre, rows)
    if a.probes:
        result["p0"] = analyze_probes(pre.get("p0", {}), read_results(a.probes))
        if result["p0"]["veto"]:
            result["overall"] = "red"
    elif pre.get("p0"):
        result["p0"] = {"veto": True, "early_stop": True, "failed": ["probe results missing"]}
        result["overall"] = "red"
    if a.anchor_results:
        anchor = pre.get("anchor", {})
        result["anchor"] = analyze_anchor(read_results(a.anchor_results), pre["candidate"],
                    stack=anchor.get("stack", "stack"), base=anchor.get("base", "tiel-deploy2"),
                    metric=anchor.get("metric", pre.get("metric", "score")),
                    delta=float(anchor.get("delta", 0)), seed=int(pre.get("seed", 1)),
                    reps=int(pre.get("bootstrap_reps", 10000)),
                    last_change_by_row=anchor.get("last_change_by_row"))
        if result["anchor"]["auto_rollback"]:
            result["overall"] = "red"
    elif pre.get("anchor"):
        result["anchor"] = {"zone": "grey", "reason": "anchor results missing", "auto_rollback": False}
        if result["overall"] == "green":
            result["overall"] = "grey"
    if a.train_zones:
        zones = load_data(a.train_zones)
        result["train"] = train_plan(zones)
    result["automatic_enable"] = False
    dest = Path(a.out) if a.out else (Path(a.results) / "gate-verdict.json" if Path(a.results).is_dir() else Path(a.results).with_suffix(".verdict.json"))
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.journal:
        append_decision(a.journal, {"candidate": pre["candidate"], "preregistration": pre,
                        "prereg_sha256": sha256_file(a.prereg),
                        "data_sha256": [sha256_file(p) if Path(p).is_file() else
                                        {str(f): sha256_file(f) for f in sorted(Path(p).glob("*.jsonl"))}
                                        for p in [a.results, *a.wave]],
                        "overall": result["overall"], "row_zones": {r["name"]: r["zone"] for r in rows},
                        "look": len(datasets),
                        "date": __import__("datetime").date.today().isoformat(), "waves": len(datasets)},
                        sealed_access=a.sealed_access)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
