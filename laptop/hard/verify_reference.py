#!/usr/bin/env python3
"""verify_reference.py [task ...] : для каждого reference/{solution,naive,medium} копирует repo + оверлей,
кладёт hidden_test.py, гоняет его (unittest, как score2.py) и видимые тесты; печатает долю пройденных проверок.
Рабочие копии — во временном каталоге (по умолчанию $TMPDIR или /tmp)."""
import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
tasks = sys.argv[1:] or sorted(d for d in os.listdir(HERE) if re.match(r"hd\d_", d))
for t in tasks:
    td = os.path.join(HERE, t)
    for var in ("solution", "naive", "medium"):
        ov = os.path.join(td, "reference", var)
        if not os.path.isdir(ov):
            continue
        w = tempfile.mkdtemp(prefix=f"{t}-{var}-")
        shutil.copytree(os.path.join(td, "repo"), w, dirs_exist_ok=True)
        shutil.copytree(ov, w, dirs_exist_ok=True)
        shutil.copy(os.path.join(td, "hidden_test.py"), w)
        r = subprocess.run([sys.executable, "-m", "unittest", "-v", "hidden_test"], cwd=w, capture_output=True, text=True, timeout=300)
        res = re.findall(r"^(test\w+) .*\.\.\. (ok|FAIL|ERROR)", r.stderr, re.M)
        ok = sum(1 for _, s in res if s == "ok")
        failed = [n for n, s in res if s != "ok"]
        vis = None
        if os.path.isdir(os.path.join(w, "tests")):
            v = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=w, capture_output=True, text=True, timeout=300)
            vis = v.returncode == 0
        elif os.path.isdir(os.path.join(w, "test")):
            v = subprocess.run(["node", "--test"], cwd=w, capture_output=True, text=True, timeout=300)
            vis = v.returncode == 0
        print(f"{t:22s} {var:9s} hidden {ok}/{len(res)} rc={r.returncode} visible={vis}  fail={failed}")
        shutil.rmtree(w, ignore_errors=True)
