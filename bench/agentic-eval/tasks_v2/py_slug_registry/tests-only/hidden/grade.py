import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
CFG=json.loads('{"file": "slugs.py", "old": "re.sub(r\\"[^a-z0-9]+\\", \\"-\\", text)", "new": "re.sub(r\\"[^a-z0-9]+\\", \\"_\\", text)", "digest": "acb6429575a8d9d521310c11512857574858700f5fb2389a53ac57f9712c61f3", "test_path": "tests/test_public_contract.py"}')
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
