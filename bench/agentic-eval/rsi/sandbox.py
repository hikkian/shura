"""RSIL0: the editor's working folder. It holds only what an editor may see: train trajectories, the current harness files, the library
and the diff. Never hidden tests, solutions, verdicts, dev or test material (Harness2 keeps the judge out of the editor's reach)."""
import shutil
from pathlib import Path

FORBIDDEN_NAMES = {"hidden", "hidden_tests", "solution", "solution_remove", "verdict", "verdicts", "dev", "test", "splits.json"}
FORBIDDEN_PREFIXES = ("verdict", "hidden", "solution")


class LeakError(RuntimeError):
    """A forbidden item would reach the editor's folder."""


def _forbidden(rel: str, dev_test: bool = True) -> bool:
    """True when a path names something the editor must not see. `dev_test=False` for source paths on the machine, where a folder such as
    /dev/shm is not the dev split."""
    for part in (p.lower() for p in Path(rel).parts):
        if part.startswith(FORBIDDEN_PREFIXES) or part == "splits.json":
            return True
        if dev_test and part in ("dev", "test"):
            return True
        if part in ("hidden_tests", "solution_remove"):
            return True
    return False


def build(dest: Path, sources: dict) -> Path:
    """Copy `sources` ({name in the editor folder: path to copy}) into `dest`. Refuses any forbidden item before anything is copied."""
    dest = Path(dest)
    for name, src in sources.items():
        if _forbidden(name) or _forbidden(str(src), dev_test=False):
            raise LeakError(f"refused: {name} <- {src} is not allowed in an editor folder")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name, src in sources.items():
        src = Path(src)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    assert_clean(dest)
    return dest


def assert_clean(folder: Path) -> None:
    """Raise LeakError when the folder holds anything an editor must not see."""
    folder = Path(folder)
    bad = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).as_posix()
        if _forbidden(rel):
            bad.append(rel)
    if bad:
        raise LeakError("forbidden items in the editor folder: " + ", ".join(sorted(bad)[:10]))
