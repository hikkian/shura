"""LL2: build fixtures/queries.json = the hand-written queries (queries.raw.json) + generated typo queries (fixed seed).

The typo group is made by code from `plain_ru`, `plain_en`, `mixed` and `code` queries: one content word of >= 5 letters gets one slip
(two letters swapped, a letter dropped, a letter doubled, or a letter replaced by a neighbour on the keyboard). Words with digits, `_`, `-`, `/`,
`.` or `--` (identifiers, flags, paths) are left alone: a typo in an identifier is a different test. usage: python3 make_queries.py
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
SEED = 20261108
SOURCES = {"plain_ru": 14, "plain_en": 10, "mixed": 6, "code": 6}
RU_NEIGHBOURS = {"а": "ы", "е": "н", "о": "л", "и": "ш", "н": "г", "т": "ь", "с": "ч", "р": "п", "в": "ы", "к": "е", "л": "д", "д": "л",
                 "м": "и", "у": "к", "п": "а", "з": "х", "б": "ю", "я": "ч", "ч": "я", "ы": "в", "й": "ц", "ь": "т", "г": "н", "ш": "щ"}
EN_NEIGHBOURS = {"a": "s", "e": "r", "i": "o", "o": "p", "t": "y", "s": "d", "r": "t", "n": "m", "l": "k", "d": "f", "c": "v", "u": "i",
                 "h": "j", "m": "n", "p": "o", "g": "h", "b": "v", "w": "e", "y": "u", "k": "l", "f": "g", "v": "b"}
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def eligible(text):
    out = []
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        before = text[m.start() - 1] if m.start() else " "
        after = text[m.end()] if m.end() < len(text) else " "
        if len(w) >= 5 and before not in "-/_.`" and after not in "-/_.`" and not w.isupper():
            out.append(m)
    return out


def slip(word, rng):
    kind = rng.choice(["swap", "drop", "double", "neighbour"])
    i = rng.randrange(1, len(word) - 1)         # never the first letter: people rarely miss it, and prefix matching would hide the test
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    table = RU_NEIGHBOURS if re.search(r"[А-Яа-яЁё]", word) else EN_NEIGHBOURS
    near = table.get(word[i].lower())
    if near is None or near == word[i].lower():
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]       # no neighbour known for this letter: swap instead
    return word[:i] + near + word[i + 1:]


def main(argv=None):
    """Default: LL2 (queries.raw.json -> queries.json, seed 20261108). LL2b: --raw queries2.raw.json --out queries2.json --seed 20261109
    --counts plain_ru=16,plain_en=10,mixed=7,code=7."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="queries.raw.json")
    ap.add_argument("--out", default="queries.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--counts", default="")
    args = ap.parse_args(argv)
    sources = dict(SOURCES)
    if args.counts:
        sources = {k: int(v) for k, v in (kv.split("=") for kv in args.counts.split(","))}
    raw = json.loads((FIX / args.raw).read_text(encoding="utf-8"))["queries"]
    rng = random.Random(args.seed)
    out = list(raw)
    n = 0
    for group, count in sources.items():
        pool = [q for q in raw if q["group"] == group and eligible(q["text"])]
        rng.shuffle(pool)
        for q in pool[:count]:
            matches = eligible(q["text"])
            m = rng.choice(matches)
            new = slip(m.group(0), rng)
            if new.lower() == m.group(0).lower():
                continue
            n += 1
            out.append({"id": f"typo-{n:02d}", "group": "typo", "text": q["text"][:m.start()] + new + q["text"][m.end():],
                        "expected": list(q["expected"]), "from": q["id"], "slip": f"{m.group(0)} -> {new}"})
    (FIX / args.out).write_text(json.dumps({"queries": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for q in out:
        counts[q["group"]] = counts.get(q["group"], 0) + 1
    print(json.dumps(counts, ensure_ascii=False), "total", len(out))


if __name__ == "__main__":
    sys.exit(main())
