"""CI helper: turn the failures in a unittest log into GitHub annotations (readable without admin rights on the repo).
Usage: python tests/ci_annotate.py unittest.log"""
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(errors="replace").splitlines()
heads = [i for i, line in enumerate(lines) if line.startswith(("FAIL:", "ERROR:"))]
for i in heads[:8]:
    chunk = [x for x in lines[i:i + 14] if not x.startswith(("====", "----"))]
    print("::error title=" + lines[i][:90].replace(",", " ") + "::" + "%0A".join(x.replace("%", "%25") for x in chunk)[:1800])
if not heads:
    print("::error title=unittest::" + "%0A".join(x.replace("%", "%25") for x in lines[-20:])[:1800])
