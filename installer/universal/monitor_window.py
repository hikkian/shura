#!/usr/bin/env python3
"""Launcher and opt-in KDE integration for the Shura Monitor window."""
from __future__ import annotations

import configparser
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

TITLE = "Shura Monitor (compact)"
OWNER = "Shura Monitor compact (managed by shura)"
BROWSERS = (
    "brave-browser", "brave", "google-chrome", "google-chrome-stable",
    "chromium", "chromium-browser", "microsoft-edge", "vivaldi", "opera",
)
RULE_VALUES = {
    "Description": OWNER,
    "title": TITLE,
    "titlematch": "1",       # exact match
    "types": "1",            # normal window
    "above": "true", "aboverule": "2",  # Force
    "skiptaskbar": "true", "skiptaskbarrule": "2",
    "skipswitcher": "true", "skipswitcherrule": "2",
    "skippager": "true", "skippagerrule": "2",
    "desktops": "", "desktopsrule": "2",  # empty list means all desktops
    "position": "0,0", "positionrule": "3",  # Apply Initially
}


def gateway_url():
    raw = os.environ.get("SHURA_GATEWAY", "http://127.0.0.1:8080")
    parsed = urlsplit(raw)
    host = parsed.hostname or "127.0.0.1"
    if host in ("127.0.0.1", "::1"):
        host = "localhost"
    netloc = host + (f":{parsed.port}" if parsed.port else "")
    return urlunsplit((parsed.scheme or "http", netloc, "/monitor", "view=compact", ""))


def launch(argv):
    if argv == ["--full"]:
        url = gateway_url().split("/monitor?", 1)[0] + "/monitor"
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, url], start_new_session=True)
            return 0
        print(f"Open this page: {url}")
        return 0
    for browser in BROWSERS:
        executable = shutil.which(browser)
        if executable:
            subprocess.Popen([executable, f"--app={gateway_url()}", "--window-size=360,320"], start_new_session=True)
            return 0
    url = gateway_url()
    firefox = shutil.which("firefox")
    if firefox:
        opener = shutil.which("xdg-open") or firefox
        subprocess.Popen([opener, url], start_new_session=True)
        print("Firefox открывает страницу во вкладке: отдельного окна приложения и правила «поверх всех» для него нет.")
    else:
        print(f"Браузер не найден. Откройте страницу вручную: {url}")
    return 0


def _config_path():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "kwinrulesrc"


def _read_config(path):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    if path.exists():
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error as exc:
            raise RuntimeError(f"Не удалось безопасно разобрать {path}: {exc}") from exc
    return parser


def _owned_rule(parser):
    return next((section for section in parser.sections()
                 if parser.get(section, "Description", fallback="") == OWNER), None)


KEEP_BACKUPS = 5


def _backup(path):
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        index = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{index}")
            index += 1
        shutil.copy2(path, backup)
        old = sorted(path.parent.glob(f"{path.name}.bak-*"), key=lambda f: (f.stat().st_mtime_ns, f.name))
        for stale in old[:-KEEP_BACKUPS]:      # every rule change used to leave one more file in ~/.config for good
            if stale != backup:
                stale.unlink(missing_ok=True)
        return backup
    return None


def _kwrite_all(path, writes, backup):
    """Run the kwriteconfig6 calls; if one fails, put the file back as it was and say so (no half-edited rules file)."""
    try:
        for args in writes:
            _kwrite(shutil.which("kwriteconfig6"), path, *args[:3], **args[3])
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        if backup is not None:
            shutil.copy2(backup, path)
        else:
            path.unlink(missing_ok=True)
        detail = (getattr(exc, "stderr", "") or str(exc)).strip()
        print(f"Не удалось записать правила KWin ({detail}). Файл {path} возвращён в прежнее состояние.", file=sys.stderr)
        return False


def _kwrite(kwrite, path, section, key, value=None, delete=False):
    command = [kwrite, "--file", str(path), "--group", section, "--key", key]
    if delete:
        command.append("--delete")
    else:
        command.append(str(value if value is not None else ""))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _screen_position():
    doctor = shutil.which("kscreen-doctor")
    if doctor:
        try:
            result = subprocess.run([doctor, "-o"], check=True, capture_output=True, text=True, timeout=3)
            blocks = re.split(r"(?=Output:)", result.stdout)
            preferred = next((block for block in blocks if re.search(r"(?im)^\s*Primary:\s*yes", block)), "\n".join(blocks))
            match = re.search(r"Geometry:\s*(-?\d+)\s*,\s*(-?\d+)\s+(\d+)x(\d+)", preferred)
            if match:
                x, y, width, _height = map(int, match.groups())
                return f"{x + max(0, width - 360)},{y}"
        except (OSError, subprocess.SubprocessError):
            pass
    xrandr = shutil.which("xrandr")
    if xrandr:
        try:
            result = subprocess.run([xrandr, "--current"], check=True, capture_output=True, text=True, timeout=3)
            match = re.search(r"(?m)^\S+ connected primary (?:\S+ )?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", result.stdout)
            if match:
                width, _height, x, y = map(int, match.groups())
                return f"{x + max(0, width - 360)},{y}"
        except (OSError, subprocess.SubprocessError):
            pass
    return "0,0"


def _plasma_available():
    current = os.environ.get("XDG_CURRENT_DESKTOP", "") + ":" + os.environ.get("DESKTOP_SESSION", "")
    return ("plasma" in current.lower() and os.environ.get("XDG_SESSION_TYPE", "") in ("wayland", "x11")
            and shutil.which("kwriteconfig6") is not None and shutil.which("busctl") is not None)


def _rule_ids(parser):
    return [x.strip() for x in parser.get("General", "rules", fallback="").split(",") if x.strip()]


def install_rule():
    if not _plasma_available():
        print("Для этого окружения правило «поверх всех» нужно настроить вручную (требуется KDE Plasma и kwriteconfig6).")
        return 0
    path = _config_path()
    parser = _read_config(path)
    owned = _owned_rule(parser)
    ids = _rule_ids(parser)
    if owned:
        rule_id = owned
    else:
        base = "ShuraMonitorCompact"
        rule_id = base
        suffix = 2
        while rule_id in parser:
            rule_id = f"{base}{suffix}"
            suffix += 1
    backup = _backup(path)
    values = dict(RULE_VALUES)
    values["position"] = _screen_position()
    if rule_id not in ids:
        ids.append(rule_id)
    writes = [(rule_id, key, value, {}) for key, value in values.items()]
    writes += [("General", "rules", ",".join(ids), {}), ("General", "count", str(len(ids)), {})]
    if not _kwrite_all(path, writes, backup):
        return 1
    subprocess.run([shutil.which("busctl"), "--user", "call", "org.kde.KWin", "/KWin", "org.kde.KWin", "reconfigure"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    print("Правило KWin установлено. Позиция задаётся при открытии, размер окном не фиксируется.")
    return 0


def remove_rule():
    if not _plasma_available():
        print("Для этого окружения правило «поверх всех» настраивается вручную; конфиги не изменены.")
        return 0
    path = _config_path()
    parser = _read_config(path)
    rule_id = _owned_rule(parser)
    if not rule_id:
        print("Правило Shura Monitor не найдено.")
        return 0
    ids = [x for x in _rule_ids(parser) if x != rule_id]
    backup = _backup(path)
    writes = [(rule_id, key, None, {"delete": True}) for key in RULE_VALUES]
    writes += [("General", "rules", ",".join(ids), {}), ("General", "count", str(len(ids)), {})]
    if not _kwrite_all(path, writes, backup):
        return 1
    subprocess.run([shutil.which("busctl"), "--user", "call", "org.kde.KWin", "/KWin", "org.kde.KWin", "reconfigure"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    print("Правило Shura Monitor удалено; остальные правила сохранены.")
    return 0


def _shura_exec():
    """The command the autostart entry runs: absolute, because the session's PATH often lacks ~/.local/bin."""
    found = shutil.which("shura") or str(Path(__file__).resolve().parents[2] / "scripts" / "shura")
    if not Path(found).is_absolute() or not os.path.exists(found):
        return "shura"
    return f'"{found}"' if any(c in found for c in ' \t"\'\\$`') else found


def autostart(mode):
    if mode not in ("on", "off"):
        print("Использование: shura monitor --autostart on|off", file=sys.stderr)
        return 2
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "shura-monitor.desktop"
    owned = path.exists() and "X-Shura-Managed=true" in path.read_text(encoding="utf-8", errors="replace")
    if mode == "off":
        if owned:
            path.unlink()
        print("Автозапуск Shura Monitor выключен.")
    else:
        if path.exists() and not owned:
            print(f"Не трогаю существующий чужой файл: {path}", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"[Desktop Entry]\nType=Application\nName=Shura Monitor\nExec={_shura_exec()} monitor\nX-Shura-Managed=true\nX-GNOME-Autostart-enabled=true\n", encoding="utf-8")
        print("Автозапуск Shura Monitor включён.")
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--install-rule"]:
        return install_rule()
    if args == ["--remove-rule"]:
        return remove_rule()
    if len(args) == 2 and args[0] == "--autostart":
        return autostart(args[1])
    if args in ([], ["--full"]):
        return launch(args)
    print("Использование: shura monitor [--full|--install-rule|--remove-rule|--autostart on|off]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
