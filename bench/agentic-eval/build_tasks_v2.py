"""Render the isolated v2 catalog. Never reads or writes the v1 tasks directory."""
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tasks_v2"
CATALOG = HERE / "families_v2" / "catalog.py"
TASK_KINDS = ("implement", "bugfix", "refactor", "tests-only")


def _catalog():
    spec = importlib.util.spec_from_file_location("agentic_eval_catalog_v2", CATALOG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FAMILIES


def _write(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _runner(lang, path):
    if lang == "python":
        return ["python3", "-B", "-m", "unittest", "discover", "-s", "hidden_tests", "-t", ".", "-q"]
    return ["node", "--test", f"hidden_tests/{path}"]


def _support_files(spec, status):
    if not spec.get("multi"):
        return {}
    if status == "start":
        return {"README.md": f"# {spec['module']}\n\nStarter project. Complete the public contract described in task.json.\n",
                "docs/design.md": "# Design notes\n\nImplementation and boundary cases: to be completed.\n"}
    return {"README.md": f"# {spec['module']}\n\nThis project exposes the contract in its entry module and keeps deterministic behavior across inputs.\n",
            "docs/design.md": "# Design notes\n\nValidation is explicit, inputs are not mutated, and edge cases are covered by the test suite.\n"}


def _visible(spec):
    if spec["lang"] == "python":
        return {"public_repro.py": f"import unittest\nfrom {spec['module']} import *\nclass PublicRepro(unittest.TestCase):\n def test_reported_case(self):\n  {spec['probe']}\n"}
    return {"public-repro.test.js": f"const test=require('node:test'),assert=require('node:assert/strict'); const {{ {spec['exports']} }}=require('./{spec['module']}');\ntest('reported behavior',()=>{{ {spec['probe']} }});\n"}


def _test_only_file(spec):
    if spec["lang"] == "python":
        return "tests/test_public_contract.py", f'''import unittest
from {spec['module']} import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        {spec['probe']}
'''
    return "tests/contract.test.js", f'''const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../{spec['module']}');
test('documents an important edge case',()=>{{ const {{ {spec['exports']} }}=api; {spec['probe']} }});
'''


def _grade_script(spec, file_name, old, new, digest, test_path):
    cfg = json.dumps({"file": file_name, "old": old, "new": new, "digest": digest, "test_path": test_path})
    if spec["lang"] == "python":
        return f'''import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
CFG=json.loads({cfg!r})
def run(root):
    env={{**os.environ,"PYTHONDONTWRITEBYTECODE":"1","NO_PROXY":"*","no_proxy":"*"}}
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
'''
    return f'''const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),os=require('node:os'),cp=require('node:child_process');
const cfg={cfg}; const root=process.cwd(), target=path.join(root,cfg.file);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function run(cwd){{const r=cp.spawnSync(process.execPath,['--test',cfg.test_path],{{cwd,encoding:'utf8',timeout:8000,env:{{...process.env,NO_PROXY:'*',no_proxy:'*'}}}});return r.status===0;}}
if(!fs.existsSync(target)||hash(target)!==cfg.digest)process.exit(2);
if(!fs.existsSync(path.join(root,cfg.test_path)))process.exit(3);
if(!run(root))process.exit(4);
const temp=fs.mkdtempSync(path.join('/dev/shm','v2-grade-'));
try{{cp.execFileSync('cp',['-a',root+'/.',temp]);const clone=temp,file=path.join(clone,cfg.file);let text=fs.readFileSync(file,'utf8');if(text.split(cfg.old).length!==2)process.exit(5);fs.writeFileSync(file,text.replace(cfg.old,cfg.new));if(run(clone))process.exit(6);}}finally{{fs.rmSync(temp,{{recursive:true,force:true}});}}
'''


def render(spec, out=OUT):
    name, lang, file_name = spec["name"], spec["lang"], spec["file"]
    source = spec["solution"]
    old, new = spec["mutation"]
    if source.count(old) != 1:
        raise ValueError(f"{name}: regression anchor must match once")
    buggy = source.replace(old, new)
    api_prompt = spec["api"]
    common = f"\n\nUse only the standard library. Keep the public API and specified error behavior. Do not add dependencies or access the network."
    if spec.get("long"):
        api_prompt += (f"\n\nThis is a long-horizon project (target {spec['long']} deliberate work steps). Work through the contract, boundaries, integration points, regression cases, and final verification in small checkpoints. "
                       "The deliverable spans a public facade, an implementation module, and project documentation; keep them consistent.")
    if spec.get("multi"):
        api_prompt += "\n\nThis change crosses multiple files (3-6): keep the public facade, implementation, and supporting project notes aligned."
    hidden = {"test_contract.py" if lang == "python" else "contract.test.js": spec["tests"]}
    if lang == "python":
        hidden["__init__.py"] = ""
    if lang == "python":
        repro_cmd = "python3 -B -m unittest -q public_repro"
        export_names = None
    else:
        repro_cmd = "node --test public-repro.test.js"
        export_names = spec.get("export", "")
    project_hidden = dict(hidden)
    if spec.get("multi"):
        if lang == "python":
            project_hidden["test_project_docs.py"] = f'''from pathlib import Path
import unittest
class ProjectDocs(unittest.TestCase):
 def test_supporting_files_are_completed(self):
  readme=Path("README.md").read_text()
  design=Path("docs/design.md").read_text()
  self.assertIn("{spec['module']}", readme)
  self.assertNotIn("Starter project", readme)
  self.assertIn("Validation is explicit", design)
'''
        else:
            project_hidden["contract.test.js"] += f'''\ntest('supporting project files are completed',()=>{{const fs=require('node:fs');const readme=fs.readFileSync('README.md','utf8'),design=fs.readFileSync('docs/design.md','utf8');assert.ok(readme.includes('{spec['module']}'));assert.ok(!readme.includes('Starter project'));assert.ok(design.includes('Validation is explicit'));}});\n'''
    variants = {}

    # New implementation from an intentionally incomplete project.
    variants["implement"] = {
        "start": {file_name: spec["stub"], **_support_files(spec, "start")}, "solution": {file_name: source, **_support_files(spec, "solution")}, "remove": [],
        "hidden": project_hidden, "prompt": "Implement the requested behavior in the existing module." + common,
        "kind": "implement", "difficulty": spec["difficulty"], "visible_cmd": None,
    }

    # A regression with a visible symptom plus broader hidden coverage.
    variants["bugfix"] = {
        "start": {file_name: buggy, **_visible(spec), **_support_files(spec, "start")}, "solution": {file_name: source, **_support_files(spec, "solution")}, "remove": [],
        "hidden": project_hidden, "prompt": f"The existing module has a user-visible regression. Reproduce it with `{repro_cmd}`, fix the implementation (leave the reproduction untouched), then verify the public contract." + common,
        "kind": "bugfix", "difficulty": spec["difficulty"], "visible_cmd": (["python3", "-B", "-m", "unittest", "-q", "public_repro"] if lang == "python" else ["node", "--test", "public-repro.test.js"]),
    }

    # Structural refactor. Contract tests stay the same; the hidden check also requires a facade/core split.
    if lang == "python":
        package = spec["module"]
        split_solution = {f"{package}/core.py": source, f"{package}/__init__.py": f"from .core import *  # stable public facade\n", **_support_files(spec, "solution")}
        split_hidden = dict(project_hidden)
        split_hidden["test_layout.py"] = f'''from pathlib import Path
import unittest
class Layout(unittest.TestCase):
 def test_public_package_is_split(self):
  self.assertTrue(Path("{package}/__init__.py").is_file())
  self.assertTrue(Path("{package}/core.py").is_file())
  self.assertFalse(Path("{file_name}").exists())
'''
    else:
        package = spec["module"]
        split_solution = {f"{package}/core.js": source, f"{package}/index.js": f"module.exports = require('./core');\n", **_support_files(spec, "solution")}
        split_hidden = dict(project_hidden)
        split_hidden["contract.test.js"] += f'''\ntest('public package is split',()=>{{const fs=require('node:fs');assert.ok(fs.existsSync('{package}/index.js'));assert.ok(fs.existsSync('{package}/core.js'));assert.ok(!fs.existsSync('{file_name}'));}});\n'''
    variants["refactor"] = {
        "start": {file_name: source, **_support_files(spec, "start")}, "solution": split_solution, "remove": [file_name],
        "hidden": split_hidden,
        "prompt": f"Refactor the working module into a small package named `{package}` with a stable public facade and a separate core implementation. Preserve every existing behavior and error contract; do not change the API. Keep the change reviewable across 3 files." + common,
        "kind": "refactor", "difficulty": "hard" if spec.get("multi") else "medium", "visible_cmd": None,
    }

    # Tests-only task: the grader mutates the implementation after the submitted tests pass.
    if lang == "python":
        import hashlib
        digest = hashlib.sha256(source.encode()).hexdigest()
        test_path, public_test = _test_only_file(spec)
        grade = _grade_script(spec, file_name, old, new, digest, test_path)
        test_hidden = {"grade.py": grade}
        test_hidden["__init__.py"] = ""
        test_cmd = ["python3", "-B", "hidden_tests/grade.py"]
        ref_solution = {test_path: public_test}
    else:
        import hashlib
        digest = hashlib.sha256(source.encode()).hexdigest()
        test_path, public_test = _test_only_file(spec)
        grade = _grade_script(spec, file_name, old, new, digest, test_path)
        test_hidden = {"grade.cjs": grade}
        test_cmd = ["node", "hidden_tests/grade.cjs"]
        ref_solution = {test_path: public_test}
    variants["tests-only"] = {
        "start": {file_name: source}, "solution": ref_solution, "remove": [], "hidden": test_hidden,
        "prompt": "Add focused regression tests under the existing test folder. Do not edit production files. Cover the contract boundary shown by the API and ensure the test suite would catch a realistic regression." + common,
        "kind": "tests-only", "difficulty": "medium", "visible_cmd": None, "test_cmd": test_cmd,
    }

    for variant, data in variants.items():
        task_dir = out / name / variant
        if task_dir.exists():
            shutil.rmtree(task_dir)
        _write(task_dir / "start", data["start"])
        _write(task_dir / "solution", data["solution"])
        _write(task_dir / "hidden", data["hidden"])
        task = {
            "id": f"{name}-{variant}", "family": name, "kind": data["kind"], "lang": lang,
            "difficulty": data["difficulty"], "prompt": api_prompt + "\n\n" + data["prompt"],
            "test_cmd": data.get("test_cmd", _runner(lang, "contract.test.js")),
            "timeout_s": 900, "solution_remove": data["remove"], "visible_cmd": data["visible_cmd"],
            "files_changed_target": "3-6" if spec.get("multi") or variant == "refactor" else "1-2",
            "long_horizon_steps": spec.get("long", 0) if spec.get("long") else 0,
        }
        (task_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(out=OUT):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    families = _catalog()
    family_dir = HERE / "families_v2"
    (family_dir / "__init__.py").write_text('"""Separate v2 task family package."""\n', encoding="utf-8")
    for spec in families:
        family_file = family_dir / f"{spec['name']}.py"
        family_file.write_text(
            '"""Catalog entry for an independent v2 task family."""\n'
            'from .catalog import FAMILIES\n'
            f'FAMILY = next(item for item in FAMILIES if item["name"] == {spec["name"]!r})\n',
            encoding="utf-8",
        )
        render(spec, out)
    return len(families) * len(TASK_KINDS)


if __name__ == "__main__":
    print(f"rendered {build()} v2 tasks in {OUT}")
