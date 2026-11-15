"""INSTALL1 acceptance rule: the installer never runs sudo. Any line that mentions sudo must only print it
(a command for the user to copy), and no line may have sudo in command position.
Run: python3 -m unittest tests.test_installer_no_sudo   (SHURA_SETUP_SH=path checks another copy of setup.sh)"""
import ast
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP = Path(os.environ.get("SHURA_SETUP_SH", ROOT / "setup.sh"))
PRINT_WORDS = re.compile(r"\b(echo|printf|info|say|warn|die|ask|print|log)\b")
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"|\'[^\']*\'')
COMMENT = re.compile(r"(^|\s)#.*$")
# sudo where a command starts: line start, or after ; && || | ( $( { `
CMD_POS = re.compile(r"(^|;|&&|\|\||\||\(|\)|\{|`)\s*(env\s+\S+=\S+\s+)*sudo\b")


def code_files():
    return [SETUP] + sorted((ROOT / "installer").rglob("*.py"))


def sudo_lines(path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if "sudo" in line:
            yield n, line


def strip_comment(line, py):
    if py:
        return re.sub(r"\s#.*$", "", line) if not line.lstrip().startswith("#") else ""
    return COMMENT.sub("", line)


# Python: a call that would run sudo is a string "sudo ..." (or a list whose first item is "sudo") given to a process
# runner: subprocess.* / os.system / os.popen / Popen. Strings in print calls, docstrings, comments and permission
# rules such as {"sudo *": "ask"} are not calls and are ignored.
RUNNERS = {"run", "call", "check_call", "check_output", "Popen", "system", "popen", "spawnv", "execv", "execvp"}


def _is_sudo_text(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"(^|[\s;&|(`])sudo(\s|$)", node.value)


def python_sudo_calls(text):
    """Line numbers where sudo is given to a process runner (a planted call must be found; a print must not)."""
    tree = ast.parse(text)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in RUNNERS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if _is_sudo_text(arg):
                hits.append(node.lineno)
            if isinstance(arg, (ast.List, ast.Tuple)) and any(_is_sudo_text(e) for e in arg.elts):
                hits.append(node.lineno)
    return hits


class NoSudo(unittest.TestCase):
    def test_python_never_runs_sudo(self):
        bad = []
        for path in sorted((ROOT / "installer").rglob("*.py")):
            for n in python_sudo_calls(path.read_text(encoding="utf-8")):
                bad.append(f"{path.relative_to(ROOT)}:{n}")
        self.assertEqual(bad, [], "sudo given to a process runner:\n" + "\n".join(bad))

    def test_planted_sudo_call_is_found(self):
        planted = [
            'subprocess.run(["sudo", "dnf", "install", "-y", "x"])',
            'os.system("sudo dnf install -y x")',
            'subprocess.check_call("sudo apt-get install x", shell=True)',
            'subprocess.Popen(["sudo", "id"])',
        ]
        for code in planted:
            self.assertTrue(python_sudo_calls(code + "\n"), code)

    def test_printed_or_stored_sudo_text_is_not_a_call(self):
        harmless = [
            'print("run this yourself: sudo dnf install -y x")',
            'say("sudo apt-get install x")',
            'RULES = {"sudo *": "ask"}',
            '"""docstring: never calls sudo"""',
            '# never calls sudo',
        ]
        for code in harmless:
            self.assertEqual(python_sudo_calls(code + "\n"), [], code)

    def test_sudo_only_in_printed_text(self):
        # shell: every line that mentions sudo must print it (a command for the user to copy)
        bad = []
        for n, line in sudo_lines(SETUP):
            code = strip_comment(line, False)
            if "sudo" not in code:
                continue
            outside = QUOTED.sub('""', code)
            if "sudo" in outside or not PRINT_WORDS.search(outside):
                bad.append(f"{SETUP.name}:{n}: {line.strip()}")
        self.assertEqual(bad, [], "sudo is executed (or not clearly printed) in setup.sh:\n" + "\n".join(bad))

    def test_no_sudo_in_command_position(self):
        bad = []
        for n, line in sudo_lines(SETUP):
            code = QUOTED.sub('""', COMMENT.sub("", line))
            if CMD_POS.search(code):
                bad.append(f"{SETUP.name}:{n}: {line.strip()}")
        self.assertEqual(bad, [], "sudo in command position:\n" + "\n".join(bad))

    def test_no_privileged_systemctl_or_package_managers_run(self):
        # without sudo the installer must not run package managers directly either (they would need root)
        bad = []
        for n, line in enumerate(SETUP.read_text().splitlines(), 1):
            code = QUOTED.sub('""', COMMENT.sub("", line))
            if re.search(r"(^|;|&&|\|\||\||\))\s*(dnf|apt-get|apt|pacman|dpkg)\s", code):
                bad.append(f"{SETUP.name}:{n}: {line.strip()}")
        self.assertEqual(bad, [], "package manager executed:\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
