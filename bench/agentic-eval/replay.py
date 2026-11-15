"""Replay recorded agent file edits in an isolated task folder.

Shell is intentionally not a general-purpose replay surface: only simple local
mkdir/rm/rmdir operations confined to the replay work folder are applied.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = Path.home() / "Desktop/AI/reports/qlev1/raw/main/results.jsonl"
DEFAULT_LOGS = DEFAULT_RESULTS.parent / "agent-logs"
IGNORED_DIRS = {"__pycache__"}


def tree_hash(root: Path) -> str:
    """Hash relative names and contents in stable order, ignoring disposable caches."""
    root = Path(root)
    digest = hashlib.sha256()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if "node_modules" in rel.parts:
            node_i = rel.parts.index("node_modules")
            tail = rel.parts[node_i + 1:]
            if ".cache" in tail:
                continue
            if path.name == "node_modules" and path.is_dir() and all(child.name == ".cache" for child in path.iterdir()):
                continue
        name = rel.as_posix().encode("utf-8", "surrogateescape")
        if path.is_symlink():
            kind, content = b"L", os.readlink(path).encode("utf-8", "surrogateescape")
        elif path.is_file():
            kind = b"F"
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            content = h.digest()
        elif path.is_dir():
            kind, content = b"D", b""
        else:
            kind, content = b"?", b""
        digest.update(kind + len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _safe_relative(raw: str) -> Path:
    """Turn a recorded tool path into a relative work-folder path."""
    p = Path(raw)
    if p.is_absolute():
        parts = p.parts
        try:
            marker = max(i for i, part in enumerate(parts) if part == "work")
        except ValueError:
            raise ValueError(f"absolute path is outside recorded work folder: {raw}")
        p = Path(*parts[marker + 1:])
    else:
        p = Path(str(p).removeprefix("./"))
    if not p.parts or any(part in ("..", "") for part in p.parts):
        raise ValueError(f"unsafe work-folder path: {raw}")
    return p


def _target(work: Path, raw: str) -> Path:
    target = work / _safe_relative(raw)
    if not target.resolve(strict=False).is_relative_to(work.resolve()):
        raise ValueError(f"path escapes replay folder: {raw}")
    return target


def _tool_events(log_path: Path, max_steps: int | None = None):
    """Tool calls of the record in order; with max_steps only those of the first max_steps model steps (FORK1b: the state
    at step k)."""
    steps = 0
    for line_no, line in enumerate(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "step_finish":
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return
        if event.get("type") == "tool_use":
            part = event.get("part") or {}
            yield line_no, part.get("tool"), part.get("state") or {}, part


def _apply_patch_fallback(content: str, patch_text: str) -> str:
    """Apply the tool's recorded unified diff when its matcher used fuzzy context."""
    lines = content.splitlines(keepends=True)
    patch_lines = patch_text.splitlines()
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    i = 0
    changed = False
    while i < len(patch_lines):
        match = hunk_re.match(patch_lines[i])
        if not match:
            i += 1
            continue
        old_start = int(match.group(1))
        i += 1
        old_block, new_block, context = [], [], []
        while i < len(patch_lines) and patch_lines[i][:1] in (" ", "+", "-"):
            line = patch_lines[i]
            body = line[1:]
            if line.startswith(" "):
                old_block.append(body)
                new_block.append(body)
                context.append((len(old_block) - 1, body))
            elif line.startswith("-"):
                old_block.append(body)
            else:
                new_block.append(body)
            i += 1
        if not old_block:
            continue
        old_count = len(old_block)
        starts = range(max(0, len(lines) - old_count + 1))
        candidates = [start for start in starts if all(
            start + offset < len(lines) and lines[start + offset].rstrip("\r\n") == expected
            for offset, expected in context
        )]
        start = old_start - 1
        if start not in candidates:
            if len(candidates) != 1:
                raise ValueError("recorded edit diff has no unique matching context")
            start = candidates[0]
        eol = "\r\n" if "\r\n" in content else "\n"
        replacement = [line + eol for line in new_block]
        if start + old_count == len(lines) and content and not content.endswith(("\n", "\r")):
            if replacement:
                replacement[-1] = replacement[-1].rstrip("\r\n")
        lines[start:start + old_count] = replacement
        changed = True
    if not changed:
        raise ValueError("recorded edit diff could not be applied")
    return "".join(lines)


def _apply_file_tool(tool: str, state: dict, work: Path, part: dict) -> None:
    data = state.get("input") or {}
    raw = data.get("filePath")
    if not isinstance(raw, str):
        raise ValueError(f"{tool} event has no filePath")
    path = _target(work, raw)
    if tool == "write":
        content = data.get("content")
        if not isinstance(content, str):
            raise ValueError("write event has no string content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return
    old, new = data.get("oldString"), data.get("newString")
    if not isinstance(old, str) or not old:
        raise ValueError("edit event has no non-empty oldString")
    if not isinstance(new, str) or not path.is_file():
        raise ValueError(f"edit target missing or malformed: {raw}")
    content = path.read_text(encoding="utf-8")
    if old not in content:
        metadata = state.get("metadata") or {}
        patch_text = metadata.get("diff") or (metadata.get("filediff") or {}).get("patch") or ""
        if not patch_text:
            raise ValueError(f"edit oldString not found and no patch was recorded: {raw}")
        updated = _apply_patch_fallback(content, patch_text)
    else:
        updated = content.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")


def _rewrite_recorded_work_path(raw: str, work: Path) -> Path:
    return _target(work, raw)


_HD_SENT = "\x00H"
_HD_RE = re.compile(r"\bcat\s+<<\s*(?:'(\w+)'|\"(\w+)\")")
SHELL_KINDS = ("mv", "cp", "sed", "write", "mkdir", "rm", "cd", "skipped")


def _scan_line(line: str):
    """Split one shell line into clauses on unquoted && and ; only.

    Each clause is (tokens, unsafe); tokens are ("w", text) or ("op", text). The
    line is never given to a shell. Anything that would need expansion (variables,
    command substitution, globs, ~, pipes, background, subshells, other
    redirections) marks the clause unsafe so that it is skipped.
    """
    clauses = []
    toks = []
    unsafe = False
    buf = []
    inword = False
    quoted = False
    n = len(line)
    i = 0

    def flush():
        nonlocal buf, inword, quoted
        if inword:
            toks.append(("w", "".join(buf)))
        buf, inword, quoted = [], False, False

    def end_clause(sep=None):
        nonlocal toks, unsafe
        flush()
        clauses.append((toks, unsafe, sep))
        toks, unsafe = [], False

    while i < n:
        c = line[i]
        if c in " \t":
            flush()
        elif c == "'":
            j = line.find("'", i + 1)
            if j < 0:
                raise ValueError("unbalanced quote")
            buf.append(line[i + 1:j])
            inword = quoted = True
            i = j
        elif c == '"':
            j = i + 1
            inword = quoted = True
            while True:
                if j >= n:
                    raise ValueError("unbalanced quote")
                d = line[j]
                if d == '"':
                    break
                if d == "\\" and j + 1 < n and line[j + 1] in '"\\$`':
                    buf.append(line[j + 1])
                    j += 2
                    continue
                if d in "$`":
                    unsafe = True
                buf.append(d)
                j += 1
            i = j
        elif c == "\\":
            if i + 1 >= n or line[i + 1] == "\n":
                raise ValueError("dangling backslash")
            buf.append(line[i + 1])
            inword = quoted = True
            i += 1
        elif c == ";":
            end_clause(";")
        elif c == "&":
            if line[i + 1:i + 2] == "&":
                end_clause("&&")
                i += 1
            else:
                flush()
                toks.append(("op", "&"))
                unsafe = True
        elif c == "|":
            if line[i + 1:i + 2] == "|":
                end_clause("||")
                i += 1
            else:
                flush()
                toks.append(("op", "|"))
                unsafe = True
        elif c == ">":
            if inword and not quoted and "".join(buf).isdigit():
                unsafe = True              # fd redirect such as 2>
            flush()
            if line[i + 1:i + 2] == ">":
                toks.append(("op", ">>"))
                i += 1
            else:
                toks.append(("op", ">"))
        elif c == "<":
            flush()
            toks.append(("op", "<"))
            unsafe = True
            while line[i + 1:i + 2] == "<":
                i += 1
        else:
            if c in "$`()\n{}*?[":
                unsafe = True
            elif c in "~#" and not inword:
                unsafe = True
            buf.append(c)
            inword = True
        i += 1
    end_clause()
    return clauses


def _confined(cwd: Path, raw: str, work: Path) -> Path:
    """Resolve a recorded path; refuse anything that ends up outside work (symlinks included)."""
    target = _rewrite_recorded_work_path(raw, work) if os.path.isabs(raw) else _target(cwd, raw)
    if not target.resolve(strict=False).is_relative_to(work.resolve()):
        raise ValueError(f"path escapes replay folder: {raw}")
    return target


def _check_inside(path: Path, work: Path) -> Path:
    if not path.resolve(strict=False).is_relative_to(work.resolve()):
        raise ValueError(f"path escapes replay folder: {path}")
    return path


def _flags_and_operands(args: list[str], allowed: set[str]):
    flags = []
    while args and args[0].startswith("-") and len(args[0]) > 1:
        if args[0] not in allowed:
            raise ValueError(f"flag not replayed: {args[0]}")
        flags.append(args[0])
        args = args[1:]
    return flags, args


def _has_symlink(root: Path) -> bool:
    """True if the folder or anything under it is a symbolic link (copying through one could leave the work folder)."""
    if root.is_symlink():
        return True
    return any(p.is_symlink() for p in root.rglob("*"))


def _regex_checked(rx: str) -> str:
    """Refuse regular expressions that can take exponential time on a file (a quantified group holding a quantifier)."""
    if len(rx) > 200 or re.search(r"\((?:[^()\\]|\\.)*[+*}]\)[+*{]", rx):
        raise ValueError("sed regex too complex to replay")
    return rx


def _sed_regex(pat: str, ere: bool) -> str:
    """Translate a simple POSIX regex to Python re; ValueError when unsure."""
    if "[:" in pat or "\\<" in pat or "\\>" in pat or "\\`" in pat or "\\'" in pat:
        raise ValueError("sed regex not replayed")
    out = []
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if c == "\\":
            if i + 1 >= n:
                raise ValueError("bad sed regex")
            d = pat[i + 1]
            i += 2
            if not ere and d in "(){}":
                out.append(d)
            elif not ere and d in "+?|":
                out.append(d)
            elif ere and d in "(){}+?|":
                out.append("\\" + d)
            elif d.isdigit() and d != "0":
                out.append("\\" + d)
            elif d in "wWsSbB.*[]^$/\\-":
                out.append("\\" + d)
            elif not d.isalnum():
                out.append(re.escape(d))
            else:
                raise ValueError("sed escape not replayed")
            continue
        if c == "[":
            j = i + 1
            if j < n and pat[j] == "^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                raise ValueError("bad bracket")
            body = pat[i + 1:j].replace("\\", "\\\\")
            out.append("[" + body + "]")
            i = j + 1
            continue
        if not ere and c in "(){}+?|":
            out.append("\\" + c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _sed_replacement(rep: str, delim: str):
    parts = []
    i = 0
    while i < len(rep):
        c = rep[i]
        if c == "\\":
            if i + 1 >= len(rep):
                raise ValueError("bad sed replacement")
            d = rep[i + 1]
            i += 2
            if d.isdigit() and d != "0":
                parts.append(("g", int(d)))
            elif d == "n":
                parts.append(("t", "\n"))
            elif d == "t":
                parts.append(("t", "\t"))
            elif d.isalnum():
                raise ValueError("sed escape not replayed")
            else:
                parts.append(("t", d))
            continue
        parts.append(("g", 0) if c == "&" else ("t", c))
        i += 1

    def render(m):
        return "".join(m.group(v) if k == "g" else v for k, v in parts)
    return render


def _split_sed_delimited(text: str, start: int, delim: str):
    """Read up to the next unescaped delim; returns (raw, index after delim)."""
    out = []
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            if text[i + 1] == delim and delim not in "/":
                out.append(text[i + 1])
            else:
                out.append(text[i:i + 2])
            i += 2
            continue
        if c == delim:
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise ValueError("unterminated sed expression")


def _sed_apply(expr: str, ere: bool, text: str) -> str:
    i = 0
    addr = None                      # predicate over (lineno, total, line)
    m = re.match(r"(\d+)(?:,(\d+))?", expr)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        addr = lambda no, line: lo <= no <= hi
        i = m.end()
    elif expr.startswith("/"):
        raw, i = _split_sed_delimited(expr, 1, "/")
        rx = re.compile(_regex_checked(_sed_regex(raw, ere)))
        addr = lambda no, line: rx.search(line) is not None
    cmd = expr[i:]
    if cmd == "d" and addr is not None:
        action = None
    elif cmd.startswith("s") and len(cmd) > 1 and not cmd[1].isalnum() and cmd[1] not in "\\ \n":
        delim = cmd[1]
        pat, j = _split_sed_delimited(cmd, 2, delim)
        rep, j = _split_sed_delimited(cmd, j, delim)
        flags = cmd[j:]
        if flags not in ("", "g"):
            raise ValueError("sed flags not replayed")
        rx = re.compile(_regex_checked(_sed_regex(pat, ere)))
        render = _sed_replacement(rep, delim)
        count = 0 if flags == "g" else 1
        action = lambda line: rx.sub(render, line, count=count)
    else:
        raise ValueError("sed command not replayed")
    out = []
    lines = text.splitlines(keepends=True)
    for no, raw_line in enumerate(lines, 1):
        body = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        nl = raw_line[len(body):]
        if addr is not None and not addr(no, body):
            out.append(raw_line)
        elif action is None:
            continue
        else:
            out.append(action(body) + nl)
    return "".join(out)


def _printf_text(args: list[str]) -> str:
    if not args:
        raise ValueError("printf without format")
    fmt, rest = args[0], args[1:]
    if fmt in ("%s\\n", "%s") and len(rest) == 1:
        return rest[0] + ("\n" if fmt.endswith("n") else "")
    if "%" in fmt or rest:
        raise ValueError("printf format not replayed")
    out = []
    i = 0
    while i < len(fmt):
        if fmt[i] == "\\":
            d = fmt[i + 1:i + 2]
            if d not in ("n", "t", "\\"):
                raise ValueError("printf escape not replayed")
            out.append({"n": "\n", "t": "\t", "\\": "\\"}[d])
            i += 2
        else:
            out.append(fmt[i])
            i += 1
    return "".join(out)


def _redirect_text(cmd_words: list[str], heredoc: str | None) -> str:
    if cmd_words == ["cat", _HD_SENT]:
        if heredoc is None:
            raise ValueError("no heredoc body")
        return heredoc
    if cmd_words and cmd_words[0] == "echo":
        args = cmd_words[1:]
        newline = True
        if args and args[0] == "-n":
            newline, args = False, args[1:]
        if len(args) > 1 or (args and ("\\" in args[0] or args[0].startswith("-"))):
            raise ValueError("echo form not replayed")      # -e/-E and any other option are not replayed
        return (args[0] if args else "") + ("\n" if newline else "")
    if cmd_words and cmd_words[0] == "printf":
        return _printf_text(cmd_words[1:])
    raise ValueError("redirect of unknown command")


def _run_clause(toks, unsafe, heredoc, cwd: Path, work: Path):
    """Replay one clause. Returns (kind, new_cwd); ValueError/OSError means skip."""
    if unsafe:
        raise ValueError("unsafe shell syntax")
    words = [t for k, t in toks if k == "w"]
    ops = [t for k, t in toks if k == "op"]
    if not words:
        raise ValueError("empty")
    if any(_HD_SENT in w for w in words) and not (words[:2] == ["cat", _HD_SENT] and ops):
        raise ValueError("unsupported heredoc form")
    if ops:
        if len(ops) != 1 or ops[0] not in (">", ">>") or toks[-2] != ("op", ops[0]) or toks[-1][0] != "w":
            raise ValueError("redirection not replayed")
        text = _redirect_text([t for k, t in toks[:-2]], heredoc)
        target = _confined(cwd, toks[-1][1], work)
        if target.is_dir() or not target.parent.is_dir():
            raise ValueError("write target not usable")
        with open(target, "a" if ops[0] == ">>" else "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return "write", cwd
    op = words[0]
    if op == "cd" and len(words) == 2:
        new = cwd if words[1] == "." else _confined(cwd, words[1], work)
        if not new.is_dir():
            raise ValueError("recorded cd target does not exist in replay")
        return "cd", new
    if op == "mkdir" and len(words) >= 3 and words[1] == "-p":
        for raw in words[2:]:
            _confined(cwd, raw, work).mkdir(parents=True, exist_ok=True)
        return "mkdir", cwd
    if op in ("rm", "rmdir"):
        args = words[1:]
        recursive = force = False
        while args and args[0].startswith("-") and args[0] != "--":
            f = args[0]
            if f.startswith("--"):
                if f == "--recursive":
                    recursive = True
                elif f == "--force":
                    force = True
                else:
                    raise ValueError("rm option not replayed")
            elif re.fullmatch(r"-[rRf]+", f):
                recursive |= any(ch in "rR" for ch in f[1:])
                force |= "f" in f[1:]
            else:
                raise ValueError("rm option not replayed")
            args = args[1:]
        if args and args[0] == "--":
            args = args[1:]
        if not args:
            raise ValueError("no removal target")
        for raw in args:
            target = _confined(cwd, raw, work)
            if target == work:
                raise ValueError("refusing to remove replay root")
            if not target.exists() and not target.is_symlink() and not force:
                raise ValueError("rm: no such file (the command would fail)")
            if target.is_dir() and recursive:
                shutil.rmtree(target, ignore_errors=True)
            elif op == "rmdir":
                target.rmdir()
            else:
                target.unlink(missing_ok=True)
        return "rm", cwd
    if op in ("mv", "cp"):
        allowed = {"-f"} if op == "mv" else {"-r", "-R", "-f", "-rf", "-fr", "-Rf", "-fR"}
        flags, operands = _flags_and_operands(words[1:], allowed)
        if len(operands) != 2:
            raise ValueError("mv/cp needs exactly SRC DST")
        src = _confined(cwd, operands[0], work)
        dst = _confined(cwd, operands[1], work)
        if not os.path.lexists(src) or src.resolve() == work.resolve():
            raise ValueError("source missing or is the replay root")
        if dst.is_dir():
            dst = _check_inside(dst / src.name, work)
        if op == "mv":
            shutil.move(str(src), str(dst))
        elif src.is_dir():
            if not any("r" in f.lower() for f in flags):
                raise ValueError("cp of a directory needs -r")
            if dst.resolve().is_relative_to(src.resolve()):
                raise ValueError("cp into itself")
            if _has_symlink(src) or (dst.is_dir() and _has_symlink(dst)):
                raise ValueError("cp -r of a tree with symlinks is not replayed")
            shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
        return op, cwd
    if op == "sed":
        args = words[1:]
        ere = False
        has_i = False
        expr = None
        while args and args[0].startswith("-") and len(args[0]) > 1:
            a = args.pop(0)
            if a == "-i":
                has_i = True
            elif a in ("-E", "-r"):
                ere = True
            elif a == "-e":
                if not args:
                    raise ValueError("sed -e without expression")
                expr = args.pop(0)
            else:
                raise ValueError(f"sed flag not replayed: {a}")
        if not has_i:
            raise ValueError("sed without -i is not a mutation")
        if expr is None:
            if not args:
                raise ValueError("sed without expression")
            expr = args.pop(0)
        if len(args) != 1:
            raise ValueError("sed needs exactly one file")
        target = _confined(cwd, args[0], work)
        if not target.is_file():
            raise ValueError("sed target missing")
        updated = _sed_apply(expr, ere, target.read_text(encoding="utf-8"))
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
        return "sed", cwd
    raise ValueError("command not replayed")


def replay_safe_shell_detail(command: str, work: Path) -> dict:
    """Parse (shlex-free scanner, never a shell) and replay confined file mutations.

    Replayed: mkdir -p, rm/rmdir, cd, mv, cp [-r], sed -i with a simple s/// or
    addressed expression, echo/printf/quoted-heredoc written with > or >>.
    Everything else (tests, pipes, substitutions, unknown commands, paths outside
    the folder or through links) is skipped. Returns {"done", "skipped", "reason",
    "counts"}; counts has one number per kind in SHELL_KINDS.
    """
    counts = {k: 0 for k in SHELL_KINDS}

    def result(reason=None):
        done = sum(v for k, v in counts.items() if k not in ("skipped", "cd"))
        return {"done": done, "skipped": counts["skipped"],
                "reason": reason or ("confined filesystem mutations only" if done else "no approved filesystem mutation"),
                "counts": counts}

    command = command.strip()
    if "\x00" in command or "\r" in command:
        counts["skipped"] += 1
        return result("control characters not replayed")
    lines = command.split("\n")
    units = []                       # (line, heredoc body)
    if len(lines) == 1:
        units.append((lines[0], None))
    else:
        i = 0
        found = False
        while i < len(lines):
            line = lines[i]
            m = _HD_RE.search(line)
            if not m:
                if "<<" in line:             # another heredoc form: its body must not run as commands
                    counts["skipped"] += 1
                    return result("unsupported heredoc")
                units.append((line, None))
                i += 1
                continue
            delim = m.group(1) or m.group(2)
            end = next((j for j in range(i + 1, len(lines)) if lines[j] == delim), None)
            if end is None:
                counts["skipped"] += 1
                return result("unterminated heredoc")
            body = "".join(l + "\n" for l in lines[i + 1:end])
            units.append((line[:m.start()] + "cat " + _HD_SENT + line[m.end():], body))
            found = True
            i = end + 1
        if not found:
            counts["skipped"] += 1
            return result("shell syntax/redirection not replayed")
    cwd = work
    for line, body in units:
        if not line.strip():
            continue
        try:
            clauses = _scan_line(line)
        except ValueError:
            counts["skipped"] += 1
            if len(units) > 1:
                break
            continue
        last_ok = True                        # result of the previous clause (a skipped clause counts as failed)
        for k, (toks, unsafe, _sep) in enumerate(clauses):
            if not toks:
                continue
            before = clauses[k - 1][2] if k else ";"
            if (before == "&&" and not last_ok) or (before == "||" and last_ok):
                counts["skipped"] += 1        # bash would not run this clause after the previous result
                continue
            try:
                kind, cwd = _run_clause(toks, unsafe, body, cwd, work)
                counts[kind] += 1
                last_ok = True
            except Exception:                 # any failure (also OverflowError, RecursionError) skips this clause only
                counts["skipped"] += 1
                last_ok = False
    return result()


def replay_safe_shell(command: str, work: Path) -> tuple[int, int, str]:
    """(replayed mutation clauses, skipped clauses, explanation); see replay_safe_shell_detail."""
    r = replay_safe_shell_detail(command, work)
    return r["done"], r["skipped"], r["reason"]


def replay_log(log_path: Path, work: Path, max_steps: int | None = None) -> dict:
    """Replay the completed write/edit tool calls and approved shell mutations (of the first max_steps model steps)."""
    applied = rerun = skipped = 0
    shell = {k: 0 for k in SHELL_KINDS}
    errors = []
    for line_no, tool, state, part in _tool_events(log_path, max_steps):
        status = state.get("status")
        if status not in ("completed", "error"):
            skipped += 1
            continue
        if tool in ("write", "edit"):
            if status != "completed" or not isinstance(state.get("output"), str):
                skipped += 1
                continue
            try:
                _apply_file_tool(tool, state, work, part)
                applied += 1
            except (OSError, ValueError) as exc:
                errors.append(f"line {line_no} {tool}: {exc}")
                skipped += 1
        elif tool == "bash":
            command = (state.get("input") or {}).get("command")
            if status != "completed" or not isinstance(command, str):
                skipped += 1
                continue
            detail = replay_safe_shell_detail(command, work)
            rerun += detail["done"]
            skipped += detail["skipped"]
            for k, v in detail["counts"].items():
                shell[k] += v
        else:
            skipped += 1
    return {"applied": applied, "rerun": rerun, "skipped": skipped, "errors": errors, "shell": shell}


def select_runs(rows: list[dict], logs_dir: Path, seed: int = 20261113, per_outcome: int = 5) -> list[dict]:
    """Seeded, stratified selection: five passing/failing rows, diversifying families."""
    eligible = []
    for row in rows:
        if row.get("arm") != "base" or row.get("tests_total") is None or row.get("tests_passed") is None:
            continue
        filename = f"{row['task']}__{row['arm']}__{row['rep']}.agent.log"
        if (Path(logs_dir) / filename).is_file():
            eligible.append(row)
    rng = random.Random(seed)
    chosen, families = [], set()
    for resolved in (False, True):
        candidates = [r for r in eligible if bool(r.get("resolved")) is resolved]
        rng.shuffle(candidates)
        for row in candidates:
            if sum(bool(r.get("resolved")) is resolved for r in chosen) >= per_outcome:
                break
            if row.get("family") not in families:
                chosen.append(row)
                families.add(row.get("family"))
        for row in candidates:
            if sum(bool(r.get("resolved")) is resolved for r in chosen) >= per_outcome:
                break
            if row not in chosen:
                chosen.append(row)
                families.add(row.get("family"))
    if len(chosen) != per_outcome * 2:
        raise ValueError(f"need at least {per_outcome} passing and failing eligible runs; found {len(chosen)}")
    if len(families) < 5:
        raise ValueError(f"seeded selection has only {len(families)} families")
    return chosen


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def repo_root(path: Path) -> Path:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, timeout=60)
    return Path(out.stdout.strip())


def hidden_from_rev(task_hidden: Path, rev: str, dest_root: Path) -> Path:
    """Export the hidden-test folder as it was at git revision `rev` into dest_root and return that folder.

    The hidden tests must be the version the recorded run used; this takes it from git instead of the work tree.
    Only regular files are exported; names that escape the folder are refused."""
    task_hidden = Path(task_hidden).resolve()
    repo = repo_root(task_hidden)
    rel = task_hidden.relative_to(repo).as_posix()
    blob = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev, "--", rel],
                          capture_output=True, check=True, timeout=120).stdout
    out = dest_root / "hidden-at-rev"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not name.as_posix().startswith(rel + "/"):
                raise ValueError(f"unexpected archive member: {member.name}")
            target = out / name.relative_to(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tf.extractfile(member).read())
            count += 1
    if count == 0:
        raise ValueError(f"no hidden files at {rev}: {rel}")
    return out


def run_probe(selected: list[dict], logs_dir: Path, tasks_root: Path, scratch_root: Path, hidden_rev: str | None = None) -> list[dict]:
    """Restore selected run ends and score them using each task's normal hidden-test command.

    The score is harness.run_tests (the task's own test_cmd) followed by harness.soft_score. With hidden_rev the hidden
    tests are taken from that git revision, and each outcome records whether they equal the work-tree copy."""
    # Imported lazily so tree hashing and the unit tests do not import the full
    # experiment harness, which has optional runtime integrations.
    sys.path.insert(0, str(HERE))
    import harness

    tasks = {t["id"]: t for t in harness.all_tasks(tasks_root)}
    scratch_root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for index, row in enumerate(selected, 1):
        task = tasks.get(row["task"])
        if task is None:
            raise ValueError(f"task not found: {row['task']}")
        run_name = f"{row['task']}__{row['arm']}__{row['rep']}.agent.log"
        run_dir = scratch_root / f"{index:02d}-{row['task']}-{row['rep']}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        work = run_dir / "work"
        start = task["dir"] / "start"
        shutil.copytree(start, work) if start.is_dir() else work.mkdir()
        replay_stats = replay_log(Path(logs_dir) / run_name, work)
        hidden = task["dir"] / "hidden"
        hidden_same = None
        if hidden_rev:
            pinned = hidden_from_rev(hidden, hidden_rev, scratch_root)
            hidden_same = tree_hash(pinned) == tree_hash(hidden)
            hidden = pinned
        actual_ok, tail = harness.run_tests(work, hidden, task["test_cmd"], timeout=task.get("test_timeout_s", 120))
        actual = harness.LAST_TEST_RUN.get("soft") or {}
        outcomes.append({"task": row["task"], "family": row["family"], "recorded_passed": row["tests_passed"],
                         "recorded_total": row["tests_total"], "replayed_passed": actual.get("tests_passed"),
                         "replayed_total": actual.get("tests_total"), "resolved": bool(row["resolved"]),
                         "match": actual.get("tests_passed") == row["tests_passed"] and actual.get("tests_total") == row["tests_total"],
                         "actions_applied": replay_stats["applied"], "actions_rerun": replay_stats["rerun"],
                         "actions_skipped": replay_stats["skipped"], "errors": replay_stats["errors"],
                         "test_ok": actual_ok, "test_tail": tail, "hidden_rev": hidden_rev or "", "hidden_same_as_worktree": hidden_same})
        shutil.rmtree(run_dir)
    return outcomes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--tasks", type=Path, default=HERE / "tasks_v3")
    ap.add_argument("--scratch", type=Path, default=Path("/dev/shm/FORK1"))
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--count-per-result", type=int, default=5)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--hidden-rev", default=None, help="git revision whose hidden tests are used (default: work tree)")
    a = ap.parse_args(argv)
    scratch = a.scratch
    if not os.access(scratch.parent, os.W_OK):
        scratch = HERE / ".scratch/FORK1"
    rows = select_runs(read_jsonl(a.results), a.logs, a.seed, a.count_per_result)
    try:
        outcomes = run_probe(rows, a.logs, a.tasks, scratch, a.hidden_rev)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    payload = {"seed": a.seed, "hidden_rev": a.hidden_rev or "", "selected": [{k: r.get(k) for k in ("task", "family", "resolved", "tests_passed", "tests_total", "rep")} for r in rows],
               "outcomes": outcomes, "matches": sum(x["match"] for x in outcomes), "total": len(outcomes)}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["matches"] >= 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
