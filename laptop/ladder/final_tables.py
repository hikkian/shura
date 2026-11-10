#!/usr/bin/env python3
"""final_tables.py — все таблицы для итоговых отчётов одним запуском.

Только чтение: ~/Desktop/AI/shura-report/results/*.jsonl. Разбор строк и доля
проверок скрытого теста берутся из ladder_score.py (load_arm / enrich), не
переписаны. Пишет два файла рядом с этим скриптом:
  final_tables.md  — все таблицы (markdown)
  final_data.json  — те же числа структурно (ключи по arm и по задаче)

Учитываются строки, записанные на момент запуска: идущие прогоны дают меньшее n.
Нет данных -> «—» в md, null в json.
"""
import datetime
import glob
import json
import os
import statistics
import sys

sys.dont_write_bytecode = True  # не создавать __pycache__ рядом с ladder_score.py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_score as ls  # noqa: E402

RESULTS = ls.RESULTS
OUT_MD = os.path.join(HERE, "final_tables.md")
OUT_JSON = os.path.join(HERE, "final_data.json")
DASH = "—"

MODES = ["fast", "low", "medium", "high", "xhigh", "max"]
LADDER = {m: f"arm_ornith5_{m}_lad" for m in MODES}
HEAVY = {"hd1", "hd2", "hd3", "hd4", "hd5", "hd6"}
HD5 = {"hd5"}
NEW = {"h1", "h2", "h3", "h4", "h5"}
OLD5 = {"t3", "t12", "t14", "t16", "t19"}
TEMPLATE_ARMS = [
    ("arm_ornith4_xhigh_ho", "ornith4 xhigh_ho"),
    ("arm_ornith5_xhigh_ho", "ornith5 xhigh_ho"),
    ("arm_ornithsharp_xhigh_ho", "ornithsharp xhigh_ho"),
    ("arm_ornith6_xhigh_ho", "ornith6 xhigh_ho"),
    ("arm_ornith7_xhigh_ho", "ornith7 xhigh_ho"),
]
CODEX = "codex"
CODEX_FINAL = "codex.final"
OUR_FINAL5 = ["arm_ornith4_medium.final", "arm_ornith4_high.final",
              "arm_ornith4_xhigh.final", "arm_ornith4_max.final"]


# ---------- загрузка (через ladder_score) ----------

_cache = {}


def load(arm):
    """-> {'recs': [...], 'limit': int} или None, если файла results/<arm>.jsonl нет."""
    if arm not in _cache:
        rows = ls.load_arm(arm)
        if rows is None:
            _cache[arm] = None
        else:
            # лимит «в срок» зависит от слова max в имени: суффикс .final его ломает
            recs, limit = ls.enrich(arm.split(".")[0], rows)
            _cache[arm] = {"recs": list(recs.values()), "limit": limit, "raw": rows}
    return _cache[arm]


def recs_of(arm):
    d = load(arm)
    return d["recs"] if d else []


def tid(task):
    return task.split("_", 1)[0]


def sel(recs, ids=None, exclude=None):
    out = recs
    if ids is not None:
        out = [r for r in out if tid(r["task"]) in ids]
    if exclude is not None:
        out = [r for r in out if tid(r["task"]) not in exclude]
    return out


def task_names(ids, arms):
    names = set()
    for a in arms:
        for r in sel(recs_of(a), ids=ids):
            names.add(r["task"])
    return sorted(names)


# ---------- агрегаты ----------

def rep_means(recs):
    by = {}
    for r in recs:
        by.setdefault(r["rep"], []).append(r["frac"])
    return {str(k): statistics.mean(v) for k, v in sorted(by.items())}


def span(rm):
    """Размах (min, max) средней доли по повторам; только если повторов 2+."""
    if len(rm) < 2:
        return None
    vals = list(rm.values())
    return [min(vals), max(vals)]


def agg(recs):
    n = len(recs)
    if n == 0:
        return {"n": 0, "passed": None, "mean_frac": None, "timeouts": None,
                "median_secs": None, "fallback": False, "rep_means": {}, "rep_span": None}
    rm = rep_means(recs)
    return {
        "n": n,
        "passed": sum(bool(r["pass"]) for r in recs),
        "mean_frac": statistics.mean(r["frac"] for r in recs),
        "timeouts": sum(bool(r["timeout"]) for r in recs),
        "median_secs": statistics.median(r["secs"] for r in recs),
        "fallback": any(r["star"] for r in recs),
        "rep_means": rm,
        "rep_span": span(rm),
    }


def limit_agg(arm):
    d = load(arm)
    if d is None:
        return {"n": 0, "rc124": None, "rc124_pass": None, "pass_in_time": None, "limit": None}
    recs = d["recs"]
    return {
        "n": len(recs),
        "rc124": sum(bool(r["timeout"]) for r in recs),
        "rc124_pass": sum(bool(r["timeout"] and r["pass"]) for r in recs),
        "pass_in_time": sum(bool(r["in_time"]) for r in recs),
        "limit": d["limit"],
    }


# ---------- форматирование ----------

def dash(x):
    return DASH if x is None else str(x)


def fp(x, star=False):
    return DASH if x is None else f"{100 * x:.1f}%" + ("*" if star else "")


def fmed(x):
    return DASH if x is None else f"{x:.0f}"


def fspan(s):
    return DASH if s is None else f"{100 * s[0]:.1f}–{100 * s[1]:.1f}%"


def frep(rm):
    return " · ".join(f"rep{k} {100 * v:.1f}%" for k, v in rm.items()) or DASH


def cell_frac_n(a):
    if a["n"] == 0:
        return DASH
    return f"{fp(a['mean_frac'], a['fallback'])} (n={a['n']})"


def cell_summary(a):
    """прошло/всего · ср. доля · таймауты · медиана"""
    if a["n"] == 0:
        return DASH
    return (f"{a['passed']}/{a['n']} · {fp(a['mean_frac'], a['fallback'])} · "
            f"TO {a['timeouts']} · {fmed(a['median_secs'])} с")


def row_full(label, arm, a):
    return [label, arm, dash(a["n"] or None), dash(a["passed"]) if a["n"] else DASH,
            fp(a["mean_frac"], a["fallback"]), fmed(a["median_secs"]),
            dash(a["timeouts"]) if a["n"] else DASH, frep(a["rep_means"]), fspan(a["rep_span"])]


def md_table(headers, body):
    return ls.md_table(headers, body)


def main():
    now = datetime.datetime.now().isoformat(timespec="seconds")
    js = {"generated_at": now, "source": RESULTS,
          "notes": {
              "share": "ср. доля = среднее по прогонам; прогон = passed/total проверок скрытого теста",
              "span": "размах = min и max средней доли по повторам (среднее по задачам внутри повтора)",
              "star": "fallback: доля по полю pass (1/0), вывод hidden-теста не разобран",
              "missing": "null в json / «—» в md: данных нет",
          }}
    md = [f"# Итоговые таблицы бенча",
          "",
          f"Сгенерировано: {now}. Источник: `{RESULTS}/*.jsonl` (только чтение). "
          "Учтены строки, записанные на момент запуска: идущие прогоны дают меньшее n.",
          "",
          "Обозначения: «—» нет данных; `*` доля по полю pass (вывод hidden-теста не разобран); "
          "ср. доля — среднее по прогонам (passed/total проверок скрытого теста); "
          "«по повторам» — средняя доля по каждому повтору (среднее по задачам); "
          "размах — min–max этих средних (виден шум; есть при 2+ повторах).",
          ""]

    # ----- 1. Лесенка -----
    t1_modes = {}
    md += ["## 1. Лесенка режимов (hd1–hd6)", ""]
    body = []
    for m in MODES:
        arm = LADDER[m]
        recs = recs_of(arm)
        a = agg(sel(recs, ids=HEAVY))
        b = agg(sel(recs, ids=HEAVY, exclude=HD5))
        h5 = agg(sel(recs, ids=HD5))
        t1_modes[m] = {"arm": arm, "file_exists": load(arm) is not None,
                       "all_hd1_hd6": a, "without_hd5": b, "hd5": h5}
        body.append([m, arm, dash(a["n"]) if a["n"] else DASH,
                     a["passed"] if a["n"] else DASH,
                     fp(a["mean_frac"], a["fallback"]),
                     fp(b["mean_frac"], b["fallback"]),
                     a["timeouts"] if a["n"] else DASH,
                     fmed(a["median_secs"]),
                     frep(a["rep_means"]), fspan(a["rep_span"])])
    md += [md_table(["режим", "arm", "n", "прошло целиком", "ср. доля (hd1–hd6)",
                     "ср. доля без hd5", "таймауты (rc=124)", "медиана, с",
                     "по повторам", "размах по повторам"], body), ""]
    md += ["### hd5 по режимам (всегда упирается в лимит)", ""]
    body = []
    for m in MODES:
        h5 = t1_modes[m]["hd5"]
        body.append([m, h5["n"] or DASH, h5["passed"] if h5["n"] else DASH,
                     fp(h5["mean_frac"], h5["fallback"]),
                     h5["timeouts"] if h5["n"] else DASH, fmed(h5["median_secs"])])
    md += [md_table(["режим", "n", "прошло", "ср. доля", "таймауты (rc=124)", "медиана, с"], body), ""]
    js["table1_ladder"] = {"modes": t1_modes}

    # ----- 2. Задача x режим -----
    heavy_names = task_names(HEAVY, [LADDER[m] for m in MODES])
    t2_tasks = {}
    body = []
    for name in heavy_names:
        row = [name]
        t2_tasks[name] = {}
        for m in MODES:
            a = agg(sel(recs_of(LADDER[m]), ids={tid(name)}))
            t2_tasks[name][m] = a
            row.append(cell_frac_n(a))
        body.append(row)
    span_row = ["размах по повторам (все hd1–hd6)"]
    t2_span = {}
    for m in MODES:
        s = agg(sel(recs_of(LADDER[m]), ids=HEAVY))["rep_span"]
        t2_span[m] = s
        span_row.append(fspan(s))
    body.append(span_row)
    md += ["## 2. Задача × режим (средняя доля, n в скобках)", "",
           md_table(["задача"] + MODES, body), ""]
    js["table2_task_x_mode"] = {"tasks": t2_tasks, "mode_rep_span": t2_span}

    # ----- 3. Шаблоны на h1..h5 (xhigh) -----
    arms3 = [a for a, _ in TEMPLATE_ARMS] + [CODEX]
    labels3 = [lab for _, lab in TEMPLATE_ARMS] + ["codex (Luna)"]
    names3 = task_names(NEW, arms3)
    t3 = {"arms": {}, "tasks": names3}
    body = []
    for name in names3:
        row = [name]
        for arm in arms3:
            a = agg(sel(recs_of(arm), ids={tid(name)}))
            row.append(cell_summary(a))
        body.append(row)
    total_row = ["**все h1–h5**"]
    span3_row = ["размах по повторам (ср. по задачам)"]
    for arm, lab in zip(arms3, labels3):
        a = agg(sel(recs_of(arm), ids=NEW))
        t3["arms"][arm] = {"label": lab, "file_exists": load(arm) is not None,
                           "tasks": {name: agg(sel(recs_of(arm), ids={tid(name)})) for name in names3},
                           "total_h1_h5": a}
        total_row.append(cell_summary(a))
        span3_row.append(fspan(a["rep_span"]))
    body += [total_row, span3_row]
    md += ["## 3. Шаблоны на новых задачах h1–h5 (xhigh)", "",
           "В ячейке: прошло/всего · ср. доля · таймауты (TO) · медиана времени.", "",
           md_table(["задача"] + labels3, body), ""]
    js["table3_h1_h5_templates"] = t3

    # ----- 4. Luna (Codex) против нас -----
    t4 = {}
    # 4a: hd1..hd6, codex 2 повтора против лесенки
    rows4a = []
    rows_a = {}
    for m in MODES:
        a = agg(sel(recs_of(LADDER[m]), ids=HEAVY))
        rows_a[m] = a
        rows4a.append(row_full(f"ladder {m}", LADDER[m], a))
    ca = agg(sel(recs_of(CODEX), ids=HEAVY))
    rows_a["codex"] = ca
    rows4a.append(row_full("codex (Luna)", CODEX, ca))
    md += ["## 4. Luna (Codex) против нас", "",
           "### 4a. hd1–hd6: codex (2 повтора) против режимов лесенки", "",
           md_table(["режим", "arm", "n", "прошло", "ср. доля", "медиана, с", "таймауты",
                     "по повторам", "размах"], rows4a), ""]
    t4["hd1_hd6"] = {"rows": rows_a}

    # 4b: h1..h5, codex против ornith5
    rows4b = []
    rows_b = {}
    for arm, lab in [(CODEX, "codex (Luna)"), ("arm_ornith5_xhigh_ho", "ornith5 xhigh_ho")]:
        a = agg(sel(recs_of(arm), ids=NEW))
        rows_b[arm] = a
        rows4b.append(row_full(lab, arm, a))
    md += ["### 4b. h1–h5: codex против ornith5 (xhigh_ho)", "",
           md_table(["вариант", "arm", "n", "прошло", "ср. доля", "медиана, с", "таймауты",
                     "по повторам", "размах"], rows4b), ""]
    t4["h1_h5"] = {"rows": rows_b}

    # 4c: старый набор 19 задач, codex.final; 5 задач для наших .final
    old_all = agg(recs_of(CODEX_FINAL))
    old_5_codex = agg(sel(recs_of(CODEX_FINAL), ids=OLD5))
    rows4c = [["codex.final, все 19 задач", CODEX_FINAL,
               f"{old_all['passed']}/{old_all['n']}" if old_all["n"] else DASH,
               fp(old_all["mean_frac"], old_all["fallback"])],
              ["codex.final, 5 задач t3,t12,t14,t16,t19", CODEX_FINAL,
               f"{old_5_codex['passed']}/{old_5_codex['n']}" if old_5_codex["n"] else DASH,
               fp(old_5_codex["mean_frac"], old_5_codex["fallback"])]]
    old5_ours = {}
    for arm in OUR_FINAL5:
        a = agg(sel(recs_of(arm), ids=OLD5))
        old5_ours[arm] = a
        rows4c.append([f"наши, 5 задач t3,t12,t14,t16,t19", arm,
                       f"{a['passed']}/{a['n']}" if a["n"] else DASH,
                       fp(a["mean_frac"], a["fallback"])])
    md += ["### 4c. Старый набор (19 задач, codex.final) и 5 общих задач", "",
           md_table(["набор", "arm", "прошло/всего", "ср. доля"], rows4c), ""]
    t4["old_set_19"] = {CODEX_FINAL: old_all}
    t4["old_set_5_tasks"] = {"codex": old_5_codex, "ours": old5_ours}
    js["table4_codex_vs_us"] = t4

    # ----- 5. В срок / после таймаута -----
    rows5 = []
    arms5 = sorted(os.path.basename(p)[:-len(".jsonl")]
                   for p in glob.glob(os.path.join(RESULTS, "arm_ornith4_*.final.jsonl")))
    t5_arms = {}
    tot = {"n": 0, "rc124": 0, "rc124_pass": 0, "pass_in_time": 0}
    for arm in arms5:
        d = limit_agg(arm)
        t5_arms[arm] = d
        rows5.append([arm, d["n"], d["rc124"] if d["n"] else DASH,
                      d["rc124_pass"] if d["n"] else DASH,
                      d["pass_in_time"] if d["n"] else DASH, dash(d["limit"])])
        for k in tot:
            tot[k] += d[k] or 0
    rows5.append(["**всего**", tot["n"], tot["rc124"], tot["rc124_pass"], tot["pass_in_time"], "—"])
    md += ["## 5. Прошло в срок / только после таймаута (arm_ornith4_*, .final)", "",
           "«в срок» = pass и secs < лимита (как в ladder_score.py; лимит 3000 с для arm с max, иначе 1500 с). "
           "«засчитано pass при rc=124» — прогоны, которые упёрлись в таймаут и всё равно прошли.", "",
           md_table(["arm", "прогонов", "rc=124", "из них засчитано pass", "прошло в срок", "лимит, с"],
                    rows5), ""]
    js["table5_timeouts"] = {"arms": t5_arms, "total": tot}

    # ----- прогоны с долей 100% и pass=false -----
    odd = []
    for arm in sorted(_cache):
        d = load(arm)
        if not d:
            continue
        for r in d["recs"]:
            if r["frac"] == 1.0 and not r["pass"]:
                raw = d["raw"][(r["task"], r["rep"])]
                odd.append({"arm": arm, "task": r["task"], "rep": r["rep"],
                            "tests_touched": raw.get("tests_touched") or []})
    js["notes"]["full_share_but_not_pass"] = {"count": len(odd), "runs": odd}
    if odd:
        md += ["## Примечание: доля 100%, но pass=false", "",
               "В этих прогонах скрытый тест прошёл целиком (доля 100% в «ср. доля»), "
               "но прогон не засчитан как pass. Причина по данным: непустой tests_touched "
               "(правка тестовых файлов). Такие прогоны входят в средние доли, как в ladder_score.py.", "",
               md_table(["arm", "задача", "rep", "tests_touched"],
                        [[o["arm"], o["task"], o["rep"], ", ".join(o["tests_touched"]) or "—"] for o in odd]), ""]

    # ----- предупреждения -----
    if ls.WARNINGS:
        md += ["## Предупреждения (из ladder_score.py)", ""] + [f"- {w}" for w in ls.WARNINGS] + [""]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    print(f"записано: {OUT_MD}")
    print(f"записано: {OUT_JSON}")
    for w in ls.WARNINGS:
        print("предупреждение: " + w, file=sys.stderr)


if __name__ == "__main__":
    main()
