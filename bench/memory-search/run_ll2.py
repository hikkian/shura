"""LL2: memory search measurement without a model. Builds a real `lifelong_store.Store` from the synthetic fixture, runs the queries against
four search arms (u61 / tri / both / stem), computes the metrics and applies the decision rules written in PREREG.md.

usage: nice -n 19 python3 run_ll2.py [--out DIR] [--sizes 1000,5000] [--no-latency]
Databases live on tmpfs (/dev/shm): the measurement writes nothing to the SSD. No model, no GPU, no network.
"""
import argparse
import json
import math
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "gateway"))
import lifelong_store as ls  # noqa: E402

FIX = HERE / "fixtures"
ARMS = ("u61", "tri", "both", "stem", "triw")
DECIDING_ARMS = ("u61", "tri", "both")
REALISTIC = ("plain_ru", "plain_en", "code", "mixed", "typo", "case", "version")
GUARD_GROUPS = ("plain_ru", "plain_en", "code", "version")
NEGATIVE_GROUPS = ("negative", "negative_deleted")
BM25 = "bm25({t}, 4.0, 1.0, 3.0, 1.5)"
CONFIRMED_BONUS = 0.3
CHARS_PER_TOKEN = 3.0
SEED = 20261108
INDEX_BUDGET_TOKENS = 1500


# ---------------------------------------------------------------------------------------------------------------- the store
def build_store(path, store=None):
    """The fixture loaded through the production API: deleted records forgotten, chains saved then fixed, disputes saved then disputed.
    `store` lets a caller put records (padding) in first."""
    data = json.loads((FIX / "records.json").read_text(encoding="utf-8"))
    store = store or ls.Store(path, legacy_dir=None)
    for r in data["records"]:
        store.save(r["key"], r["title"], r["body"], kind=r["kind"], source=r["source"], trust=r.get("trust", "confirmed"))
        if r.get("deleted"):
            store.forget(r["key"])
    for c in data["chains"]:
        store.save(c["chain"], c["v1"]["title"], c["v1"]["body"], kind=c["kind"], source=c["source"])
        store.fix(c["chain"], title=c["v2"]["title"], body=c["v2"]["body"])
    for d in data["disputes"]:
        store.save(d["key"], d["a"]["title"], d["a"]["body"], kind=d["kind"], source=d["source"])
        store.fix(d["key"], title=d["b"]["title"], body=d["b"]["body"], relation="dispute")
    return store


def vocabulary():
    data = json.loads((FIX / "records.json").read_text(encoding="utf-8"))
    words = []
    for r in data["records"]:
        words += ls.tokens(f"{r['title']} {r['body']}")
    return sorted(set(words))


def pad_store(store, n_total, seed=SEED):
    """Synthetic padding records (random words from the fixture's vocabulary) up to `n_total` live records, for latency and index size."""
    rng = random.Random(seed)
    vocab = vocabulary()
    weights = [1.0 / (i + 1) ** 0.6 for i in range(len(vocab))]
    rng.shuffle(vocab)
    live = store.db.execute("SELECT COUNT(*) FROM memory WHERE deleted_at IS NULL").fetchone()[0]
    i = 0
    kinds = ("fact", "preference", "procedure", "note")
    while live < n_total:
        title = " ".join(rng.choices(vocab, weights, k=rng.randint(2, 4)))
        body = " ".join(rng.choices(vocab, weights, k=rng.randint(12, 30)))
        store.save(f"pad-{i}", title, body, kind=rng.choice(kinds), source="import")
        i += 1
        live += 1
    return store


def add_trigram(store):
    """The second FTS5 table (trigram) beside the production one, filled from the live records in the same way as `_index_row`."""
    db = store.db
    db.execute("DROP TABLE IF EXISTS memory_tri")
    db.execute("CREATE VIRTUAL TABLE memory_tri USING fts5(title, body, key, tags, tokenize = 'trigram')")
    db.execute("BEGIN")
    for r in db.execute("SELECT * FROM memory WHERE deleted_at IS NULL").fetchall():
        tags = " ".join(t["tag"] for t in db.execute("SELECT tag FROM tag WHERE memory_id=? ORDER BY tag", (r["id"],)))
        db.execute("INSERT INTO memory_tri(rowid, title, body, key, tags) VALUES (?,?,?,?,?)", (r["id"], r["title"], r["body"], r["key"].replace("-", " "), tags))
    db.execute("COMMIT")


# ---------------------------------------------------------------------------------------------------------------- the arms
def fts_scored(store, table, match, limit=20):
    """{key: score} from one FTS table: bm25 (title 4, body 1, key 3, tags 1.5) plus the trust bonus, as production `search()` does."""
    if not match:
        return {}
    rows = store.db.execute(
        f"SELECT m.key, m.trust, {BM25.format(t=table)} AS rank FROM {table} JOIN memory m ON m.id = {table}.rowid "
        f"WHERE {table} MATCH ? AND m.deleted_at IS NULL ORDER BY rank LIMIT ?", (match, limit)).fetchall()
    return {r["key"]: -float(r["rank"]) + (CONFIRMED_BONUS if r["trust"] == "confirmed" else 0.0) for r in rows}


def match_triw(query):
    """Exploratory arm `triw`: every word of >= 3 characters as one quoted substring (a typo inside the word cannot match)."""
    words = [w for w in ls.tokens(query)[:12] if len(w) >= 3]
    return " OR ".join(f'"{w}"' for w in words) or None


MAX_TRIGRAMS = 80


def match_tri(query):
    """Arm `tri`: OR of all the trigrams of every query word (at most 80 distinct): a word with a slip, another case or another ending still
    shares most trigrams with the stored word; bm25 sums them, so whole matches rank above partial ones."""
    grams = []
    for w in ls.tokens(query)[:12]:
        if len(w) < 3:
            continue
        for i in range(len(w) - 2):
            g = w[i:i + 3]
            if g not in grams:
                grams.append(g)
    return " OR ".join(f'"{g}"' for g in grams[:MAX_TRIGRAMS]) or None


def match_stem(query):
    words = ls.tokens(query)[:12]
    if not words:
        return None
    return " OR ".join(f'"{w[:max(4, len(w) - 2)]}"*' if len(w) >= 5 else f'"{w}"' for w in words)


def search_arm(store, arm, query, limit=5):
    """Ranked [(key, score)] for one arm; at most `limit` results."""
    if arm == "u61":
        scored = {r["key"]: r["score"] for r in store.search(query, limit=20)} if ls.fts_query(query) else {}
    elif arm == "tri":
        scored = fts_scored(store, "memory_tri", match_tri(query))
    elif arm == "triw":
        scored = fts_scored(store, "memory_tri", match_triw(query))
    elif arm == "stem":
        scored = fts_scored(store, "memory_fts", match_stem(query))
    elif arm == "both":
        a = {r["key"]: r["score"] for r in store.search(query, limit=20)} if ls.fts_query(query) else {}
        b = fts_scored(store, "memory_tri", match_tri(query))
        scored = {k: a.get(k, 0.0) + b.get(k, 0.0) - (CONFIRMED_BONUS if k in a and k in b and store._row(k)["trust"] == "confirmed" else 0.0)
                  for k in set(a) | set(b)}         # both arms already added the trust bonus; a record found by both gets it once
    else:
        raise ValueError(arm)
    return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


# ---------------------------------------------------------------------------------------------------------------- metrics
def est_tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def block_entry(store, key):
    row = store._row(key)
    rec = store.record(row)
    return f"{rec['passport_line']} {rec['title']}: {' '.join(rec['body'].split())[:240]}"


def first_hit_rank(keys, expected):
    for i, k in enumerate(keys):
        if k in expected:
            return i + 1
    return None


def tune_threshold(neg_top1, limit_share=0.10):
    """The lowest t such that the share of negatives whose best result scores above t is <= limit_share (results must score > t)."""
    cands = sorted({0.0, *neg_top1})
    for t in cands:
        if sum(1 for s in neg_top1 if s > t) / max(1, len(neg_top1)) <= limit_share:
            return t
    return cands[-1]


def two_fold_false_rate(neg_rows, limit_share=0.10):
    """Thresholds tuned on one half of the negatives (by id parity), the false-injection rate measured on the other half; averaged."""
    halves = [neg_rows[0::2], neg_rows[1::2]]
    rates = []
    for tune, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        t = tune_threshold([r["top1"] for r in tune], limit_share)
        rates.append(sum(1 for r in test if r["top1"] > t) / max(1, len(test)))
    return sum(rates) / 2


def bootstrap_diff(a, b, seed=SEED, resamples=10000):
    """Paired bootstrap of mean(b) - mean(a) over queries. Returns (diff, low, high) at 95%."""
    import random as _r
    rng = _r.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return sum(b[i] - a[i] for i in range(n)) / n, diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def evaluate_arm(store, arm, queries):
    rows = []
    for q in queries:
        res = search_arm(store, arm, q["text"])
        rows.append({"id": q["id"], "group": q["group"], "expected": q["expected"], "keys": [k for k, _ in res], "scores": [s for _, s in res],
                     "top1": res[0][1] if res else 0.0})
    return rows


def summarise(store, arm, rows):
    pos = [r for r in rows if r["group"] not in NEGATIVE_GROUPS]
    neg = [r for r in rows if r["group"] in NEGATIVE_GROUPS]
    thr = tune_threshold([r["top1"] for r in neg])
    entries = {}

    def entry_tokens(key):
        if key not in entries:
            entries[key] = est_tokens(block_entry(store, key))
        return entries[key]

    def hit(r, k, thresholded):
        keys = [key for key, s in zip(r["keys"], r["scores"]) if (s > thr or not thresholded)][:k]
        return 1 if any(x in r["expected"] for x in keys) else 0

    out = {"arm": arm, "n_pos": len(pos), "n_neg": len(neg), "threshold": thr, "groups": {}, "vectors": {}}
    for k in (1, 3, 5):
        out[f"hit@{k}"] = sum(hit(r, k, False) for r in pos) / len(pos)
    out["hit@3_thr"] = sum(hit(r, 3, True) for r in pos) / len(pos)
    out["mrr"] = sum((1.0 / first_hit_rank(r["keys"], r["expected"])) if first_hit_rank(r["keys"], r["expected"]) else 0.0 for r in pos) / len(pos)
    out["vectors"]["hit@3"] = [hit(r, 3, False) for r in pos]
    extra, total = [], []
    for r in pos:
        kept = [key for key, s in zip(r["keys"], r["scores"]) if s > thr][:3]
        extra.append(sum(entry_tokens(k) for k in kept if k not in r["expected"]))
        total.append(sum(entry_tokens(k) for k in kept))
    out["extra_tokens_pos"] = sum(extra) / len(pos)
    out["block_tokens_pos"] = sum(total) / len(pos)        # the whole injection block (all entries above the threshold, at most 3), per positive query
    neg_extra = []
    for r in neg:
        kept = [key for key, s in zip(r["keys"], r["scores"]) if s > thr][:3]
        neg_extra.append(sum(entry_tokens(k) for k in kept))
    out["extra_tokens_neg"] = sum(neg_extra) / max(1, len(neg))
    out["false_injection_rate"] = sum(1 for r in neg if r["top1"] > thr) / max(1, len(neg))
    out["false_injection_rate_2fold"] = two_fold_false_rate(neg)
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        if g in NEGATIVE_GROUPS:
            out["groups"][g] = {"n": len(sub), "injected_at_thr": sum(1 for r in sub if r["top1"] > thr) / len(sub)}
        else:
            out["groups"][g] = {"n": len(sub), "hit@1": sum(hit(r, 1, False) for r in sub) / len(sub), "hit@3": sum(hit(r, 3, False) for r in sub) / len(sub),
                                "hit@3_thr": sum(hit(r, 3, True) for r in sub) / len(sub)}
    ver = [r for r in rows if r["group"] == "version"]
    out["version_top1_ok"] = sum(1 for r in ver if r["keys"] and r["keys"][0] in r["expected"] and "~v" not in r["keys"][0]) / max(1, len(ver))
    out["deleted_leak"] = sum(1 for r in rows for k in r["keys"] if "~v" in k or k.startswith("deleted-"))
    real = [r for r in pos if r["group"] in REALISTIC]
    out["hit@3_realistic_thr"] = sum(hit(r, 3, True) for r in real) / max(1, len(real))
    out["hit@3_realistic"] = sum(hit(r, 3, False) for r in real) / max(1, len(real))
    return out


def fts_bytes(store, tables):
    total = 0
    for t in tables:
        for suffix in ("_data", "_idx", "_docsize", "_content"):
            try:
                col = "block" if suffix == "_data" else ("segid||pgno||term" if suffix == "_idx" else ("sz" if suffix == "_docsize" else "c0||c1||c2||c3"))
                row = store.db.execute(f"SELECT COALESCE(SUM(LENGTH({col})),0) FROM {t}{suffix}").fetchone()
                total += int(row[0])
            except Exception:
                pass
    return total


def latency(store, arm, queries, repeat=3):
    texts = [q["text"] for q in queries]
    for t in texts:                    # warm-up pass
        search_arm(store, arm, t)
    samples = []
    for _ in range(repeat):
        for t in texts:
            t0 = time.perf_counter()
            search_arm(store, arm, t)
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"p50_ms": round(samples[len(samples) // 2], 3), "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 3), "n": len(samples)}


def index_capacity():
    """Compact index size (tokens) of the production `render_index()` at N = 50, 100, 200 records of the fixture's style."""
    base = Path(tempfile.mkdtemp(prefix="ll2-idx-", dir="/dev/shm" if Path("/dev/shm").exists() else None))
    out = {}
    try:
        for n in (50, 100, 200):
            store = build_store(base / f"idx{n}.sqlite")
            live = [r["key"] for r in store.db.execute("SELECT key FROM memory WHERE deleted_at IS NULL ORDER BY id")]
            if n < len(live):
                for key in live[n:]:
                    store.forget(key)
            else:
                pad_store(store, n)
            text = store.render_index()
            out[str(n)] = {"live": store.db.execute("SELECT COUNT(*) FROM memory WHERE deleted_at IS NULL").fetchone()[0],
                           "chars": len(text), "tokens_est": est_tokens(text), "fits_budget": est_tokens(text) <= INDEX_BUDGET_TOKENS,
                           "compact_titles_only": n > ls.MAX_RECORDS_IN_INDEX}
            store.close()
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------------------------------------------- decisions
def decide_trigram(summ, lat1000, eps=1e-9):
    base = summ["u61"]
    res = {}
    for arm in ("tri", "both"):
        s = summ[arm]
        diff, lo, hi = bootstrap_diff(base["vectors"]["hit@3"], s["vectors"]["hit@3"])
        guards = {g: s["groups"][g]["hit@3"] - base["groups"][g]["hit@3"] for g in GUARD_GROUPS}
        cond = {
            "1_gain_ge_5pp_and_ci_low_gt_0": diff >= 0.05 - eps and lo > 0,
            "2_no_group_loses_more_than_2pp": all(d >= -0.02 - eps for d in guards.values()),
            "3_extra_tokens_not_higher_than_10pct": s["extra_tokens_pos"] <= base["extra_tokens_pos"] * 1.10 + 1.0,
            "4_p95_le_20ms_at_1000": lat1000.get(arm, {}).get("p95_ms") is not None and lat1000[arm]["p95_ms"] <= 20.0,    # not measured = no win
        }
        res[arm] = {"diff_hit3": diff, "ci": [lo, hi], "group_deltas": guards, "conditions": cond, "wins": all(cond.values())}
    winners = [a for a in ("tri", "both") if res[a]["wins"]]
    if not winners:
        verdict = "u61"
    elif len(winners) == 1:
        verdict = winners[0]
    else:
        verdict = "both" if summ["both"]["hit@3"] - summ["tri"]["hit@3"] >= 0.02 - eps else "tri"
    return verdict, res


def decide_inject(summ, chosen, index):
    s = summ[chosen]
    H = s["hit@3_realistic_thr"]
    F = s["false_injection_rate_2fold"]
    fits100 = index["100"]["fits_budget"]
    if H >= 0.85 and F <= 0.10:
        verdict = "B_optional" if fits100 else "B_default_candidate"
    else:
        verdict = "A_only"
    return verdict, {"H": H, "F_2fold": F, "index_fits_at_100": fits100, "chosen_arm": chosen}


def decide_embeddings(summ, chosen):
    groups = summ[chosen]["groups"]
    worst = {g: v["hit@3"] for g, v in groups.items() if g not in NEGATIVE_GROUPS and g != "paraphrase"}
    ok = all(v >= 0.85 for v in worst.values())
    return ("not_needed_now" if ok else "needed_later"), {"hit@3_by_group_without_paraphrase": worst,
                                                          "paraphrase_hit@3": groups.get("paraphrase", {}).get("hit@3")}


# ---------------------------------------------------------------------------------------------------------------- main
def fmt(x, pct=True):
    return f"{x * 100:.1f}" if pct else f"{x:.2f}"


def report_tables(summ, trig, trig_res, lat, sizes, index, inject, emb, stem_note, inject_all):
    lines = ["# LL2 raw summary (machine-written)", ""]
    lines.append("| arm | hit@1 | hit@3 | hit@5 | MRR | hit@3 at threshold | threshold | extra tokens (pos) | block tokens (pos) | extra tokens (neg) | false inj. | false inj. 2-fold | version top-1 | deleted leak |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        s = summ[a]
        lines.append(f"| {a} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} | {fmt(s['hit@5'])} | {s['mrr']:.3f} | {fmt(s['hit@3_thr'])} | {s['threshold']:.2f} | "
                     f"{s['extra_tokens_pos']:.1f} | {s['block_tokens_pos']:.1f} | {s['extra_tokens_neg']:.1f} | {fmt(s['false_injection_rate'])} | {fmt(s['false_injection_rate_2fold'])} | "
                     f"{fmt(s['version_top1_ok'])} | {s['deleted_leak']} |")
    groups = sorted({g for a in ARMS for g in summ[a]["groups"] if g not in NEGATIVE_GROUPS})
    lines += ["", "hit@3 by group, %:", "", "| group | n | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for g in groups:
        lines.append(f"| {g} | {summ['u61']['groups'][g]['n']} | " + " | ".join(fmt(summ[a]['groups'][g]['hit@3']) for a in ARMS) + " |")
    lines += ["", "Negatives injected at the operating threshold, %:", ""]
    for g in NEGATIVE_GROUPS:
        lines.append(f"- {g} (n={summ['u61']['groups'][g]['n']}): " + ", ".join(f"{a} {fmt(summ[a]['groups'][g]['injected_at_thr'])}" for a in ARMS))
    lines += ["", "Decision 1 (trigram): **" + trig + "**", ""]
    for a, r in trig_res.items():
        lines.append(f"- {a}: Δhit@3 = {r['diff_hit3'] * 100:+.1f} pp, 95% CI [{r['ci'][0] * 100:+.1f}; {r['ci'][1] * 100:+.1f}], "
                     f"conditions {json.dumps(r['conditions'])}, group deltas {json.dumps({k: round(v * 100, 1) for k, v in r['group_deltas'].items()})}")
    lines += ["", "Latency (tmpfs, one thread):", "", "| records | arm | p50, ms | p95, ms |", "|---|---|---|---|"]
    for n, per_arm in lat.items():
        for a in ARMS:
            lines.append(f"| {n} | {a} | {per_arm[a]['p50_ms']} | {per_arm[a]['p95_ms']} |")
    lines += ["", "FTS index size, bytes:", "", "| records | unicode61 | trigram | ratio |", "|---|---|---|---|"]
    for n, v in sizes.items():
        lines.append(f"| {n} | {v['u61']} | {v['tri']} | {v['tri'] / max(1, v['u61']):.1f}x |")
    lines += ["", "Compact index (production `render_index()`), tokens at 3 chars/token, budget 1500:", ""]
    for n, v in index.items():
        lines.append(f"- N={n}: {v['tokens_est']} tokens, fits budget: {v['fits_budget']}, titles-only mode: {v['compact_titles_only']}")
    lines += ["", "Exploratory arms against the same four conditions (not part of the verdict):", ""]
    for ex, d in stem_note.items():
        lines.append(f"- {ex}: Δhit@3 = {d['diff_hit3'] * 100:+.1f} pp, CI [{d['ci'][0] * 100:+.1f}; {d['ci'][1] * 100:+.1f}], conditions {json.dumps(d['conditions'])}")
    lines += ["", "Decision 2 / 3 for every arm (the verdict below uses the arm Decision 1 chose; PREREG says 'the best arm', which is ambiguous):", ""]
    for a in ARMS:
        r = inject_all[a]
        lines.append(f"- {a}: inject {r['inject'][0]} (H {r['inject'][1]['H'] * 100:.1f}%, F 2-fold {r['inject'][1]['F_2fold'] * 100:.1f}%), embeddings {r['embeddings']}")
    lines += ["", f"Decision 2 (auto-help): **{inject[0]}** {json.dumps(inject[1])}", f"Decision 3 (embeddings): **{emb[0]}** {json.dumps(emb[1], ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--sizes", default="1000,5000")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = json.loads((FIX / "queries.json").read_text(encoding="utf-8"))["queries"]
    tmp = Path(tempfile.mkdtemp(prefix="ll2-", dir="/dev/shm" if Path("/dev/shm").exists() else None))
    try:
        store = build_store(tmp / "fixture.sqlite")
        add_trigram(store)
        rows = {a: evaluate_arm(store, a, queries) for a in ARMS}
        summ = {a: summarise(store, a, rows[a]) for a in ARMS}
        lat, sizes = {}, {}
        if not args.no_latency:
            sizes_list = [int(x) for x in args.sizes.split(",") if x]
            n_fix = store.db.execute("SELECT COUNT(*) FROM memory WHERE deleted_at IS NULL").fetchone()[0]
            lat[str(n_fix)] = {a: latency(store, a, queries) for a in ARMS}
            sizes[str(n_fix)] = {"u61": fts_bytes(store, ["memory_fts"]), "tri": fts_bytes(store, ["memory_tri"])}
            for n in sizes_list:
                big = pad_store(build_store(tmp / f"big{n}.sqlite"), n)
                add_trigram(big)
                lat[str(n)] = {a: latency(big, a, queries) for a in ARMS}
                sizes[str(n)] = {"u61": fts_bytes(big, ["memory_fts"]), "tri": fts_bytes(big, ["memory_tri"])}
                big.close()
        lat1000 = lat.get("1000", {})
        trig, trig_res = decide_trigram(summ, lat1000)
        index = index_capacity()
        inject = decide_inject(summ, trig, index)
        emb = decide_embeddings(summ, trig)
        per_arm = {a: {"inject": decide_inject(summ, a, index), "embeddings": decide_embeddings(summ, a)[0]} for a in ARMS}
        stem_note = {}
        for ex in ("stem", "triw"):
            lt = lat1000.get(ex, {})
            detail = decide_trigram({**summ, "tri": summ[ex], "both": summ[ex]}, {"tri": lt, "both": lt})[1]["tri"]
            stem_note[ex] = {"wins_same_rules": detail["wins"], "diff_hit3": detail["diff_hit3"], "ci": detail["ci"], "conditions": detail["conditions"],
                             "group_deltas": detail["group_deltas"]}
        result = {"summary": {a: {k: v for k, v in summ[a].items() if k != "vectors"} for a in ARMS}, "trigram_decision": trig, "trigram_detail": trig_res,
                  "stem_note": stem_note, "latency": lat, "index_bytes": sizes, "index_capacity": index, "inject": {"verdict": inject[0], **inject[1]},
                  "embeddings": {"verdict": emb[0], **emb[1]}, "decisions_2_3_per_arm": {a: {"inject": v["inject"][0], "H": v["inject"][1]["H"], "embeddings": v["embeddings"]} for a, v in per_arm.items()}, "n_queries": len(queries)}
        (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        text = report_tables(summ, trig, trig_res, lat, sizes, index, inject, emb, stem_note, per_arm)
        (out_dir / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
