#!/usr/bin/env python3
"""EXAM200d: checks of the exam v3r1 set, without rewriting its own log files.

  validate  rebuild tasks_v3r1 into a scratch folder (same code as build_v3r1.py), compare the hidden-test hashes
            with manifest-v3r1.json, and run every task: reference must pass, start must fail (bwrap, as build_v3r1).
  run       the same for the Russian slice (build_v3r1_ru.py) and for the two control sets, when their layout allows.

Runs are meant to go through: systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19 ionice -c3 ...
Scratch: /dev/shm/EXAM200d/ (deleted at the end by the operator).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_v3r1 as b1  # noqa: E402

SCRATCH = Path("/dev/shm/EXAM200d")


def hidden_digest(task_dir: Path) -> str:
    digest = hashlib.sha256()
    for p in sorted((task_dir / "hidden").rglob("*")):
        if p.is_file():
            digest.update(p.relative_to(task_dir).as_posix().encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()


def cmd_validate(args) -> int:
    out = SCRATCH / "tasks-v3r1"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    log = b1.build(out)
    manifest = json.loads((HERE / "manifest-v3r1.json").read_text(encoding="utf-8"))["tasks"]
    want = {t["id"]: t["hidden_sha256"] for t in manifest}
    mismatch = [r["id"] for r in log if want.get(r["id"]) != r["hidden_sha256"]]
    print(f"built {len(log)} tasks; hidden hash mismatches vs manifest: {len(mismatch)} {mismatch[:10]}")
    n, bad = b1.validate(out, args.only)
    print(f"validated {n} tasks; problems: {len(bad)}")
    for x in bad:
        print("  ", json.dumps(x, ensure_ascii=False))
    (SCRATCH / "validate.json").write_text(json.dumps({"n": n, "bad": bad, "hash_mismatch": mismatch},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if bad or mismatch else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--only", default=None)
    sub.add_parser("tests-only")
    sub.add_parser("sets")
    a = ap.parse_args(argv)
    if a.cmd == "tests-only":
        return cmd_tests_only(a)
    if a.cmd == "sets":
        return cmd_sets(a)
    return cmd_validate(a)




# ---- tests-only tasks (step 2 of EXAM200d) -------------------------------------------------------------------------
import ast  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

TASKS = HERE.parent / "tasks_v3"


def load_cfg(task_dir: Path) -> dict:
    """The injected mutation of a tests-only grader: {file, old, new, digest}. Works for grade.cjs and grade.py."""
    hidden = task_dir / "hidden"
    cjs, py = hidden / "grade.cjs", hidden / "grade.py"
    if cjs.exists():
        text = cjs.read_text(encoding="utf-8")
        i = text.index("cfg=") + len("cfg=")
        obj, _end = json.JSONDecoder().raw_decode(text, i)
        return obj
    if py.exists():
        text = py.read_text(encoding="utf-8")
        m = re.search(r"CFG\s*=\s*json\.loads\((['\"].*?['\"])\)\n", text, re.S)
        if not m:
            raise ValueError("no CFG in grade.py")
        return json.loads(ast.literal_eval(m.group(1)))
    raise ValueError("no grader in hidden/")


def run_bwrap_cmd(work: Path, cmd: list, timeout: int) -> int:
    argv = ["nice", "-n", "19", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", str(work), str(work), "--unshare-net", "--chdir", str(work)] + cmd
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def make_work(parts: list, hidden_dir: Path | None, tmp: Path) -> Path:
    """Copies the given folders in order (later ones overwrite earlier) into a fresh work folder; hidden goes to hidden_tests."""
    work = tmp / "work"
    shutil.copytree(parts[0], work)
    for extra in parts[1:]:
        shutil.copytree(extra, work, dirs_exist_ok=True)
    if hidden_dir is not None:
        shutil.copytree(hidden_dir, work / "hidden_tests", dirs_exist_ok=True)
    return work


def check_tests_only(task_dir: Path) -> dict:
    tj = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    fam = task_dir.parent.name
    res = {"family": fam, "id": tj["id"], "idea": tj.get("idea"), "lang": tj.get("lang"), "part": None}
    try:
        cfg = load_cfg(task_dir)
    except Exception as exc:  # the grader format is unknown
        res.update({"outcome": "не применимо", "reason": f"формат грейдера: {exc}"})
        return res
    if "old" not in cfg or "new" not in cfg:
        res.update({"outcome": "не применимо", "reason": "грейдер не подставляет одну строку (other format: " +
                    ",".join(sorted(cfg)[:4]) + ")"})
        return res
    # sanity A: the reference tests of this task (solution/) must catch the injected mutation (grader exits 0)
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="tests-only-") as t:
        tmp = Path(t)
        work = make_work([task_dir / "start", task_dir / "solution"], task_dir / "hidden", tmp)
        res["grader_on_reference_rc"] = run_bwrap_cmd(work, tj["test_cmd"], min(tj.get("timeout_s", 300), 300))
    # contract check B: the hidden contract tests of the sibling implement task, on its reference, then on the mutant
    impl = TASKS / fam / "implement"
    if not (impl / "task.json").exists():
        res.update({"outcome": "не применимо", "reason": "нет соседней задачи implement"})
        return res
    itj = json.loads((impl / "task.json").read_text(encoding="utf-8"))
    ihid = impl / "hidden"
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="contract-") as t:
        tmp = Path(t)
        ref_parts = [impl / "start"] + ([impl / "solution"] if (impl / "solution").exists() else [])
        base = make_work(ref_parts, ihid, tmp)
        for name in itj.get("solution_remove", []):
            target = base / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        ref_rc = run_bwrap_cmd(base, itj["test_cmd"], min(itj.get("timeout_s", 300), 300))
        res["contract_on_reference_rc"] = ref_rc
        if ref_rc != 0:
            res.update({"outcome": "не применимо", "reason": f"контрактные тесты не проходят на эталоне implement (rc {ref_rc})"})
            return res
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="mutant-") as t:
        tmp = Path(t)
        mut = make_work(ref_parts, ihid, tmp)
        p = mut / cfg["file"]
        if not p.is_file():
            res.update({"outcome": "не применимо", "reason": f"нет файла {cfg['file']} в эталоне implement"})
            return res
        text = p.read_text(encoding="utf-8")
        if text.count(cfg["old"]) != 1:
            res.update({"outcome": "не применимо", "reason": "якорь подставной ошибки не найден ровно один раз"})
            return res
        p.write_text(text.replace(cfg["old"], cfg["new"], 1), encoding="utf-8")
        rc = run_bwrap_cmd(mut, itj["test_cmd"], min(itj.get("timeout_s", 300), 300))
    res["contract_on_mutant_rc"] = rc
    res["outcome"] = "поймана" if rc != 0 else "не поймана"
    res["mutation"] = {"file": cfg["file"], "old": cfg["old"], "new": cfg["new"]}
    return res


def cmd_tests_only(args) -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p.parent for p in TASKS.glob("*/tests-only/task.json"))
    out = []
    for d in dirs:
        r = check_tests_only(d)
        out.append(r)
        print(r["id"], r.get("outcome"), r.get("reason", ""), flush=True)
    (SCRATCH / "tests-only.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(json.dumps(Counter(r.get("outcome") for r in out), ensure_ascii=False))
    return 0



def cmd_sets(args) -> int:
    """Validate the Russian slice and the two control sets in place (nothing is rebuilt or written into them)."""
    import subprocess as sp
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rc_total = 0
    chk = sp.run([sys.executable, str(HERE / "build_v3r1_ru.py"), "--check"], capture_output=True, text=True)
    print("ru translations check rc", chk.returncode, chk.stdout.strip().splitlines()[0] if chk.stdout.strip() else "")
    summary = {}
    for name, folder in [("v3r1-ru", HERE / "tasks_v3r1_ru"), ("control-public-r1", HERE / "tasks_control_public_r1"),
                         ("lh-r1", HERE / "tasks_lh_r1")]:
        n, bad = b1.validate(folder, None)
        summary[name] = {"tasks": n, "problems": bad}
        print(f"{name}: validated {n} tasks; problems: {len(bad)}", flush=True)
        for x in bad:
            print("   ", json.dumps(x, ensure_ascii=False), flush=True)
        rc_total += 1 if bad else 0
    (SCRATCH / "sets.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if rc_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
