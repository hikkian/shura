import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
CFG=json.loads('{"file": "deps.py", "old": "heapq.heappop(ready)", "new": "ready.pop()", "digest": "407b7f1575194e6ea106fdd6168e0b964f80aa955e693a731ee3469672fbbed1", "test_path": "tests/test_public_contract.py"}')
def run(root):
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","NO_PROXY":"*","no_proxy":"*"}
    return subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-q"],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=8).returncode
root=Path.cwd(); target=root/CFG["file"]
if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=CFG["digest"]: sys.exit("production files must remain unchanged")
if not list((root/"tests").glob("test_*.py")): sys.exit("add discoverable tests")
if run(root)!=0: sys.exit("new tests must pass against the current implementation")
with tempfile.TemporaryDirectory(prefix="v2-grade-",dir="/dev/shm") as d:
    clone=Path(d)/"project"; shutil.copytree(root,clone,ignore=shutil.ignore_patterns("hidden_tests","__pycache__"))
    p=clone/CFG["file"]; text=p.read_text()
    if text.count(CFG["old"])!=1: sys.exit("mutation anchor mismatch")
    p.write_text(text.replace(CFG["old"],CFG["new"],1))
    if run(clone)==0: sys.exit("tests do not detect the supplied regression")
