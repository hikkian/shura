"""LL3a: work/holdout split of the control set (DESIGN-LL0 6.3).

    python3 control/splits.py build   # writes splits_control.json (public layer, in the repo) and splits_private.json (private layer, outside git)

The unit of the split is the family (one idea / one area of one repository is one family), so the holdout measures transfer, not memory.
Rule, fixed once: sha256(seed + ":" + family) -> first two bytes / 65536; below 0.40 the family is `holdout`, otherwise `work`. The result is
frozen: a family that is already in the file keeps its bucket when the file is rebuilt (new families are added by the same rule), and
`frozen_at` says when the file was first written. The holdout part is never shown to the model while rules are formulated and never
used for choosing candidates; the code that reads examples refuses a holdout task id (DESIGN-LL0 6.3).
The private file lists family names and counts only; its family names contain no commit text.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent
PUBLIC_OUT = EVAL / "splits_control.json"
PRIVATE = Path.home() / ".local/share/shura/private/control"
SEED = 20261109
HOLDOUT_BELOW = 0.40


def bucket(family):
    h = hashlib.sha256(f"{SEED}:{family}".encode()).digest()
    return "holdout" if int.from_bytes(h[:2], "big") / 65536 < HOLDOUT_BELOW else "work"


def build(out, root, valid_only):
    old = json.loads(out.read_text()) if out.is_file() else {}
    families = dict(old.get("families", {}))
    counts = {}
    for p in sorted(Path(root).glob("*/*/task.json")):
        t = json.loads(p.read_text())
        if valid_only:
            v = p.parent / "validation.json"
            if not v.is_file():
                continue
            val = json.loads(v.read_text())
            if not (val.get("ok") and val.get("commit") == t.get("commit")):  # a rebuilt or invalid task does not count
                continue
        fam = t["family"]
        families.setdefault(fam, bucket(fam))
        counts[fam] = counts.get(fam, 0) + 1
    data = {"seed": SEED, "rule": f"sha256(seed:family)[:2]/65536 < {HOLDOUT_BELOW} -> holdout, else work; whole families; frozen once written",
            "frozen_at": old.get("frozen_at") or time.strftime("%Y-%m-%d"), "families": dict(sorted(families.items())),
            "tasks": dict(sorted(counts.items()))}
    out.write_text(json.dumps(data, indent=1))
    n = sum(counts.values())
    h = sum(c for f, c in counts.items() if families[f] == "holdout")
    print(f"{out.name}: {len(counts)} families, {n} tasks, holdout {h} ({100 * h / max(n, 1):.0f} %)")


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        raise SystemExit(__doc__)
    build(PUBLIC_OUT, EVAL / "tasks_control_public", valid_only=False)
    build(PRIVATE / "splits_private.json", PRIVATE, valid_only=True)
