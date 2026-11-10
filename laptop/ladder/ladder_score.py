#!/usr/bin/env python3
"""ladder_score.py ARM[,ARM...] [--pairs]

Таблицы «уровни effort» с долей пройденных проверок скрытого теста.
Только читает данные бенча: results/<arm>.jsonl и hidden_test.py задач.

Откуда берётся доля скрытого теста
  Полного вывода скрытого теста на диске нет: score2.py пишет в jsonl только
  хвост stdout+stderr (поле hidden_err, последние 300 символов) и только для
  непройденного hidden. Разбор:
  - hidden=true: unittest прошёл целиком -> passed = total = N тестов.
  - hidden=false: total = «Ran N tests» из hidden_err (если строки нет в хвосте,
    N берётся статически: число методов test* у TestCase в hidden_test.py),
    passed = total - failures - errors из «FAILED (failures=a, errors=b)».
  - Строку разобрать нельзя -> 1/0 по полю pass, задача помечается «*».

Форматы задач
  Все hidden_test.py — unittest, вывод один и тот же (Ran N / FAILED / OK).
  t14_biglog и t19_honesty проверяют ответ регулярками по ANSWER_FILE,
  t16_mockup запускает playwright внутри unittest; отдельного разбора не нужно.

Лимит «в срок»: 1500 с; для arm, в имени которого есть слово max, 3000 с.
Если одна и та же (task, rep) встречается в jsonl несколько раз, берётся
последняя запись (она соответствует work/<arm>-<task>-<rep>).

--pairs: разница между соседними уровнями списка (в порядке аргументов),
считается только по общим (task, rep) парам.
"""
import argparse
import ast
import json
import os
import re
import statistics
import sys

SHURA = os.path.expanduser("~/Desktop/AI/shura-report")
EVAL = os.path.expanduser("~/Desktop/AI/shuracode-prompt/eval")
RESULTS = os.path.join(SHURA, "results")
TASK_DIRS = [os.path.join(EVAL, "tasks"), os.path.join(EVAL, "tasks_heldout")]

RAN_RE = re.compile(r"Ran (\d+) tests?")
FAILED_RE = re.compile(r"FAILED \(([^)]*)\)")
COUNT_RE = re.compile(r"(failures|errors)=(\d+)")
MAX_RE = re.compile(r"(^|_)max(_|$)")

WARNINGS = []
_static = {}


def static_test_count(task):
    """Число test*-методов в TestCase-классах hidden_test.py задачи (или None)."""
    if task in _static:
        return _static[task]
    n = None
    for d in TASK_DIRS:
        p = os.path.join(d, task, "hidden_test.py")
        if os.path.isfile(p):
            tree = ast.parse(open(p, encoding="utf-8").read())
            n = 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                is_tc = any((isinstance(b, ast.Attribute) and b.attr == "TestCase") or
                            (isinstance(b, ast.Name) and b.id == "TestCase") for b in node.bases)
                if is_tc:
                    n += sum(1 for f in node.body
                             if isinstance(f, ast.FunctionDef) and f.name.startswith("test"))
            break
    _static[task] = n
    return n


def hidden_counts(row):
    """-> (passed, total, starred) для одного прогона."""
    task = row["task"]
    fallback = (1, 1, True) if row["pass"] else (0, 1, True)
    if row["hidden"]:
        n = static_test_count(task)
        if n:
            return n, n, False
        return fallback
    err = row.get("hidden_err") or ""
    failed = FAILED_RE.search(err)
    if not failed:
        return fallback
    bad = sum(int(v) for _, v in COUNT_RE.findall(failed.group(1)))
    ran = RAN_RE.search(err)
    total = int(ran.group(1)) if ran else static_test_count(task)
    if ran:
        st = static_test_count(task)
        if st is not None and st != total:
            WARNINGS.append(f"{task} rep{row['rep']}: «Ran {total}» != статический счёт {st}")
    if not total:
        return fallback
    return max(total - bad, 0), total, False


def load_arm(arm):
    path = os.path.join(RESULTS, arm + ".jsonl")
    if not os.path.isfile(path):
        return None
    rows, dups = {}, 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                WARNINGS.append(f"{arm}.jsonl:{ln}: строка не JSON, пропущена")
                continue
            key = (r["task"], int(r["rep"]))
            if key in rows:
                dups += 1
            rows[key] = r
    if dups:
        WARNINGS.append(f"{arm}: дубликатов (task, rep) = {dups}, взята последняя запись")
    return rows


def enrich(arm, rows):
    limit = 3000 if MAX_RE.search(arm) else 1500
    recs = {}
    for key, r in rows.items():
        passed, total, star = hidden_counts(r)
        recs[key] = {
            "task": key[0], "rep": key[1], "pass": bool(r["pass"]),
            "frac": passed / total, "passed": passed, "total": total, "star": star,
            "secs": r["secs"], "timeout": r["rc"] == 124,
            "ru": r.get("ru_ratio"), "in_time": bool(r["pass"]) and r["secs"] < limit,
        }
    return recs, limit


def pct(x):
    return f"{100 * x:.1f}%"


def md_table(headers, body):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def summary(arm, recs, limit):
    n = len(recs)
    if n == 0:
        return [arm, 0, "—", "—", "—", "—", "—", "—", limit]
    fr = [r["frac"] for r in recs.values()]
    ok_ru = [r["ru"] for r in recs.values() if not r["timeout"] and r["ru"] is not None]
    star = "*" if any(r["star"] for r in recs.values()) else ""
    return [arm, n, sum(r["pass"] for r in recs.values()),
            sum(r["in_time"] for r in recs.values()),
            pct(statistics.mean(fr)) + star,
            f"{statistics.median(r['secs'] for r in recs.values()):.0f}",
            sum(r["timeout"] for r in recs.values()),
            f"{statistics.mean(ok_ru):.2f}" if ok_ru else "—", limit]


def task_table(arms, data):
    tasks = sorted({t for a in arms for (t, _) in data[a][0]})
    body = []
    for t in tasks:
        row = [t]
        for a in arms:
            recs = [r for (tt, _), r in data[a][0].items() if tt == t] if data[a] else []
            if not recs:
                row.append("—")
                continue
            star = "*" if any(r["star"] for r in recs) else ""
            row.append(f"{pct(statistics.mean(r['frac'] for r in recs))}{star} (n={len(recs)})")
        body.append(row)
    return md_table(["задача"] + arms, body)


def pairs_table(arms, data):
    body = []
    for a, b in zip(arms, arms[1:]):
        if not data.get(a) or not data.get(b):
            continue
        ra, rb = data[a][0], data[b][0]
        common = sorted(set(ra) & set(rb))
        if not common:
            body.append([f"{a} → {b}", 0, "—", "—", "—", "—"])
            continue
        fa = statistics.mean(ra[k]["frac"] for k in common)
        fb = statistics.mean(rb[k]["frac"] for k in common)
        pa = sum(ra[k]["pass"] for k in common)
        pb = sum(rb[k]["pass"] for k in common)
        body.append([f"{a} → {b}", len(common), pct(fa), pct(fb),
                     f"{100 * (fb - fa):+.1f}", f"{pa} → {pb}"])
    if not body:
        return "нет соседних пар из загруженных arm"
    return md_table(["переход", "общих (task, rep)", "ср. доля A", "ср. доля B",
                     "Δ доли B−A, п.п.", "pass A → B"], body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", help="список arm через запятую, например arm_ornith4_medium,arm_ornith4_high")
    ap.add_argument("--pairs", action="store_true",
                    help="добавить разницу между соседними уровнями по общим (task, rep) парам")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    data = {}
    for a in arms:
        rows = load_arm(a)
        if rows is None:
            print(f"ВНИМАНИЕ: нет файла results/{a}.jsonl, arm пропущен", file=sys.stderr)
            data[a] = None
            continue
        recs, limit = enrich(a, rows)
        data[a] = (recs, limit)
    arms_ok = [a for a in arms if data[a]]
    if not arms_ok:
        sys.exit("не найдено ни одного arm")

    print(f"## Сводка по arm (источник: {RESULTS}/<arm>.jsonl)\n")
    rows_out = [summary(a, *data[a]) for a in arms_ok]
    print(md_table(["arm", "прогонов", "pass", "pass в срок", "ср. доля скрытого теста",
                    "медиана времени, с", "таймаутов (rc=124)", "ср. ru_ratio (без таймаутов)",
                    "лимит, с"], rows_out))

    print("\n## Задача × arm: средняя доля скрытого теста (n = число прогонов)\n")
    print(task_table(arms_ok, data))

    if args.pairs:
        print("\n## Разница между уровнями (только общие (task, rep) пары)\n")
        print(pairs_table(arms_ok, data))

    if any(r["star"] for a in arms_ok for r in data[a][0].values()):
        print("\n* — доля по полю pass (1/0): вывод hidden-теста разобрать не удалось")
    if WARNINGS:
        print("\nПредупреждения:", file=sys.stderr)
        for w in WARNINGS:
            print("  " + w, file=sys.stderr)


if __name__ == "__main__":
    main()
