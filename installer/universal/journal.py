"""The installation journal: everything `shura install` creates or replaces is written down *before* it is touched, so a
failure in the middle puts the machine back (`rollback`) and `shura uninstall` removes exactly what the installer made
(and nothing else).

Format: JSON lines in `<shura home>/journal.jsonl`. The file is rewritten as a whole through a temporary file and an atomic
replace after every record, so an interrupted write leaves the previous complete journal in place, never half a line.

Entry kinds (`op`):
  created   a path that did not exist before (kind file/dir); rollback deletes it (a folder with everything in it; a folder
            made by `mkdir_p` is only removed when it is empty by then, a file nobody listed is never deleted with it)
  replaced  a path that existed (kind file/setting); `old` holds its previous bytes (base64), rollback puts them back
  snapshot  a folder that already existed; `children` are the names inside it before; rollback deletes all other children
  service   something started or enabled (a service); `action` + `name` say what to reverse. The command is built by the
            code from the closed ACTIONS list, never read from the file (an old entry with `undo` is not run)
  commit    the installation finished; `status`/`code` say how
"""
import base64
import json
import os
import re
import subprocess
from pathlib import Path

NAME = "journal.jsonl"
TMP_NAME = NAME + ".tmp"


# The only things a journal may ask us to run: action -> argv builder. Closed on purpose.
ACTIONS = {
    "user-service-disable": lambda name: ["systemctl", "--user", "disable", "--now", name],
}
SERVICE_NAME = re.compile(r"^(?:shura|ai)-[a-z0-9][a-z0-9-]{0,40}\.service$")
SHURA_SERVICES = frozenset({"ai-gateway.service"})   # the only unit the installer code creates (grep of installer/, setup.sh)
OLD_COMMAND_MSG = "journal entry with a ready command is not run: remove it by hand or reinstall"


def service_command(action, name):
    """The argv for a journal service entry, or JournalError (nothing is run) for an unknown action or a foreign name."""
    build = ACTIONS.get(action) if isinstance(action, str) else None
    if build is None:
        raise JournalError(f"unknown service action {action!r}; not run")
    if not isinstance(name, str) or not SERVICE_NAME.match(name) or name not in SHURA_SERVICES:
        raise JournalError(f"service name {name!r} is not one Shura installs; not run")
    return build(name)


class JournalError(Exception):
    pass


UNIT_NAME = "ai-gateway.service"
UNIT_MARKER = "# managed by Shura installer"


def unit_path():
    """The one place the gateway unit lives: $XDG_CONFIG_HOME/systemd/user/ai-gateway.service, else ~/.config/systemd/user/...
    The home (or the XDG folder the user named) is resolved; symlinks BELOW it are not followed, so a caller that compares this
    with `resolve()` of a path notices a symlinked ~/.config."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(os.path.abspath(xdg)).parent.resolve() / Path(xdg).name if xdg else Path.home().resolve() / ".config"
    return base / "systemd" / "user" / UNIT_NAME


def is_unit(path):
    """True when `path` really is (after resolving symlinks) the fixed unit file."""
    try:
        return Path(path).resolve() == unit_path()
    except (OSError, RuntimeError):
        return False


def is_unit_parent(path):
    """True when `path` really is a folder above the unit file (only used for removing empty folders we made)."""
    try:
        real = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    return real in unit_path().parents and real.parent != real


class Result:
    """What a rollback did. `problems` are real failures (path: why); `skipped` are things that were already gone."""
    def __init__(self):
        self.removed, self.restored, self.kept, self.skipped, self.problems = [], [], [], [], []

    @property
    def ok(self):
        return not self.problems

    def summary(self):
        text = f"removed {len(self.removed)}, restored {len(self.restored)}"
        if self.kept:
            text += f", kept {len(self.kept)}"
        if self.skipped:
            text += f", already gone {len(self.skipped)}"
        if self.problems:
            text += f", COULD NOT undo {len(self.problems)}: " + "; ".join(self.problems)
        return text


def _b64(data):
    return base64.b64encode(data).decode("ascii")


class Journal:
    def __init__(self, home, entries=None):
        self.home = Path(home)
        self.path = self.home / NAME
        self.entries = list(entries or [])

    # ---------------------------------------------------------------- reading
    @classmethod
    def load(cls, home):
        """The journal of `home`, or None when there is none (or nothing readable in it). A torn last line is ignored."""
        path = Path(home) / NAME
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        entries = []
        for line in raw.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and "op" in entry:
                entries.append(entry)
        return cls(home, entries) if entries else None

    @property
    def committed(self):
        return any(e["op"] == "commit" for e in self.entries)

    def paths(self):
        return [Path(e["path"]) for e in self.entries if "path" in e]

    # ---------------------------------------------------------------- writing
    def _flush(self):
        """Write the whole journal to a temporary file, then replace the real one: the file on disk is always complete."""
        if not self.home.is_dir():
            return                                 # the folder itself is not made yet; `mkdir_p` flushes right after
        tmp = self.home / TMP_NAME
        text = "".join(json.dumps(e, sort_keys=True) + "\n" for e in self.entries)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _add(self, entry, flush=True):
        self.entries.append(entry)
        if flush:
            self._flush()

    def record_created(self, path, kind=None):
        """`path` is about to be created (it does not exist now). Written down first. Returns False and records nothing when
        the path already exists: what was there before is not ours to delete (use `record_replaced` / `track_file`)."""
        p = Path(path)
        if p.exists() or p.is_symlink():
            return False
        self._add({"op": "created", "kind": kind or "file", "path": str(p)})
        return True

    def record_replaced(self, path, old_bytes, kind="file", mode=None):
        """`path` is about to be overwritten; `old_bytes` is what it holds now (None = it did not exist: same as created)."""
        if old_bytes is None:
            return self.record_created(path, kind)
        self._add({"op": "replaced", "kind": kind, "path": str(path), "old": _b64(old_bytes), "mode": mode})
        return True

    def track_file(self, path, kind="file"):
        """Call before writing a file: remembers its content if it exists, or that it is new."""
        p = Path(path)
        if p.is_file() and not p.is_symlink():
            self.record_replaced(p, p.read_bytes(), kind, mode=p.stat().st_mode & 0o7777)
        else:
            self.record_created(p, kind)

    def track_tree(self, path):
        """A folder that will be filled by code we do not log file by file (downloads, extracted builds). If it is new it
        is recorded as created (rollback deletes it with everything inside); if it exists, its present children are
        remembered and rollback deletes only newcomers."""
        p = Path(path)
        if p.is_dir() and not p.is_symlink():
            self._add({"op": "snapshot", "kind": "dir", "path": str(p), "children": sorted(c.name for c in p.iterdir())})
        else:
            self.record_created(p, "dir")

    def mkdir_p(self, path):
        """Create `path` and its missing parents, recording every folder that is new, top-down, before it is made."""
        p = Path(path)
        missing = []
        for q in [p, *p.parents]:
            if q.exists() or q.is_symlink():
                break
            missing.append(q)
        for q in reversed(missing):
            self._add({"op": "created", "kind": "dir", "path": str(q), "empty_only": True}, flush=False)
        p.mkdir(parents=True, exist_ok=True)
        self._flush()

    def record_service(self, name, action="user-service-disable"):
        """A service was (or is about to be) started/enabled. Only the kind of undo (`action`, from ACTIONS) and the service
        `name` are written; the command is built at undo time. Refuses an action or name that could not be undone later."""
        service_command(action, name)
        self._add({"op": "service", "kind": "service", "path": name, "action": action, "name": name})

    def commit(self, status="complete", code=0):
        """The installation is over; `shura uninstall` reads the journal from here."""
        self.entries = [e for e in self.entries if e["op"] != "commit"]
        self._add({"op": "commit", "status": status, "code": code})

    # ---------------------------------------------------------------- undoing
    def _remove(self, path, keep, res):
        p = Path(path)
        if keep and keep(p):
            res.kept.append(str(p))
            return
        if p.is_symlink() or p.is_file():
            p.unlink()
            res.removed.append(str(p))
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                self._remove(child, keep, res)
            try:
                p.rmdir()
                res.removed.append(str(p))
            except OSError:
                if not any(k == str(p) or k.startswith(str(p) + os.sep) for k in res.kept):
                    raise
        else:
            raise FileNotFoundError(str(p))

    def _in_scope(self, path, op, e):
        """Rollback touches only the Shura folder and the fixed gateway unit (and empty folders above that unit we made)."""
        try:
            real = Path(path).resolve()
            home = self.home.resolve()
        except (OSError, RuntimeError):
            return False
        if real == home or home in real.parents:
            return True
        if is_unit(path):
            return op != "snapshot"
        # empty folders made on the way (rmdir only, never recursive) above the Shura home or above the unit file
        return bool(op == "created" and e.get("empty_only") and e.get("kind") == "dir"
                    and (real in home.parents or is_unit_parent(path)))

    def _undo(self, e, keep, res, runner):
        op, path = e["op"], e.get("path")
        if op in ("created", "replaced", "snapshot") and path and not self._in_scope(path, op, e):
            raise JournalError("path outside the Shura folder: not undone")
        if op in ("created", "snapshot") and path:
            guard = Path(path).resolve()
            if guard.parent == guard or guard == Path.home().resolve() or guard in Path.home().resolve().parents:
                raise JournalError(f"refusing to delete {path} (a system folder or your home folder)")
        if op == "created":
            try:
                self._remove(path, keep, res)
            except FileNotFoundError:
                res.skipped.append(f"{path}: already gone, nothing to remove")
        elif op == "snapshot":
            p = Path(path)
            if not p.is_dir():
                res.skipped.append(f"{path}: folder is already gone, nothing to remove")
                return
            before = set(e.get("children", []))
            for child in sorted(p.iterdir()):
                if child.name not in before:
                    self._remove(child, keep, res)
        elif op == "replaced":
            p = Path(path)
            old = base64.b64decode(e["old"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".restore")
            tmp.write_bytes(old)
            if e.get("mode") is not None:
                os.chmod(tmp, e["mode"])
            os.replace(tmp, p)
            res.restored.append(str(p))
        elif op == "service":
            if "undo" in e:
                raise JournalError(OLD_COMMAND_MSG)
            cmd = service_command(e.get("action"), e.get("name"))
            proc = runner(cmd, capture_output=True, text=True, timeout=30)
            if getattr(proc, "returncode", 0):
                raise JournalError("`" + " ".join(cmd) + "` failed: " +
                                   ((proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"))
            res.removed.append(f"service {path}")

    def rollback(self, keep=None, runner=subprocess.run):
        """Undo everything in reverse order. Never raises: what cannot be undone is listed in `problems` with the reason and
        stays in the journal, so a second call tries only that. `keep(path)` -> True protects a path (uninstall keeps models).
        When nothing is left to undo the journal file itself is deleted. Safe to call again and again."""
        res = Result()
        remaining, deferred = [], []
        for e in reversed([x for x in self.entries if x["op"] != "commit"]):
            if e["op"] == "created" and e.get("empty_only"):
                deferred.append(e)                      # only after the journal file is gone (it lives in the home folder)
                continue
            try:
                self._undo(e, keep, res, runner)
            except Exception as exc:                                    # noqa: BLE001 - any failure is reported, not thrown
                reason = (exc.strerror or str(exc)) if isinstance(exc, OSError) else str(exc)
                res.problems.append(f"{e.get('path', '?')}: {type(exc).__name__}: {reason}")
                remaining.append(e)
        try:
            if remaining:
                self.entries = list(reversed(remaining + deferred))
                if self.home.is_dir():
                    self._flush()
            else:
                self.entries = []
                for name in (NAME, TMP_NAME):
                    (self.home / name).unlink(missing_ok=True)
                for e in deferred:
                    self._remove_empty(e["path"], res)
        except OSError as exc:
            res.problems.append(f"{self.path}: {type(exc).__name__}: {exc.strerror or exc}")
        return res

    def _remove_empty(self, path, res):
        if not self._in_scope(path, "created", {"empty_only": True, "kind": "dir"}):
            res.problems.append(f"{path}: JournalError: path outside the Shura folder: not undone")
            return
        try:
            Path(path).rmdir()
            res.removed.append(path)
        except FileNotFoundError:
            res.skipped.append(f"{path}: already gone, nothing to remove")
        except NotADirectoryError:
            res.skipped.append(f"{path}: not a folder any more, left alone")
        except OSError as exc:
            if Path(path).is_dir() and any(Path(path).iterdir()):
                res.skipped.append(f"{path}: not empty (holds files that are not in the journal), left alone")
            else:
                res.problems.append(f"{path}: {type(exc).__name__}: {exc.strerror or exc}")
