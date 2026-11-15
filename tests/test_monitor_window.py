import configparser
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "installer" / "universal"))
import monitor_window as app


class MonitorLauncher(unittest.TestCase):
    def test_chromium_preference_order_and_app_url(self):
        with patch.dict(os.environ, {"SHURA_GATEWAY": "http://127.0.0.1:18080"}, clear=False), \
             patch.object(app.shutil, "which", side_effect=lambda name: "/tmp/" + name if name in {"brave", "chromium"} else None), \
             patch.object(app.subprocess, "Popen") as popen:
            self.assertEqual(app.launch([]), 0)
        self.assertEqual(popen.call_args.args[0], ["/tmp/brave", "--app=http://localhost:18080/monitor?view=compact", "--window-size=360,320"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_firefox_fallback_explains_app_and_topmost_limits(self):
        with patch.object(app.shutil, "which", side_effect=lambda name: "/tmp/firefox" if name == "firefox" else None), \
             patch.object(app.subprocess, "Popen") as popen, patch("builtins.print") as output:
            app.launch([])
        self.assertEqual(popen.call_args.args[0], ["/tmp/firefox", "http://localhost:8080/monitor?view=compact"])
        self.assertIn("нет", output.call_args.args[0])

    def test_full_mode_uses_xdg_open(self):
        with patch.object(app.shutil, "which", side_effect=lambda name: "/tmp/xdg-open" if name == "xdg-open" else None), \
             patch.object(app.subprocess, "Popen") as popen:
            app.launch(["--full"])
        self.assertEqual(popen.call_args.args[0], ["/tmp/xdg-open", "http://localhost:8080/monitor"])


class MonitorIntegration(unittest.TestCase):
    def _kwin_run(self, calls):
        def run(command, **kwargs):
            calls.append(command)
            if Path(command[0]).name == "kwriteconfig6":
                path = Path(command[command.index("--file") + 1])
                group = command[command.index("--group") + 1]
                key = command[command.index("--key") + 1]
                parser = configparser.ConfigParser(interpolation=None, strict=False)
                parser.optionxform = str
                if path.exists():
                    parser.read(path, encoding="utf-8")
                if group not in parser:
                    parser.add_section(group)
                if "--delete" in command:
                    parser.remove_option(group, key)
                else:
                    parser.set(group, key, command[-1])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as out:
                    parser.write(out)
            return Mock(returncode=0, stdout="", stderr="")
        return run

    def test_install_is_idempotent_and_remove_preserves_user_rules(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            root = Path(temp)
            config = root / "kwinrulesrc"
            config.write_text("[General]\ncount=1\nrules=pip\n\n[pip]\nDescription=Picture in Picture\nabove=true\n", encoding="utf-8")
            calls = []
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": temp, "XDG_CURRENT_DESKTOP": "KDE", "DESKTOP_SESSION": "plasma", "XDG_SESSION_TYPE": "wayland"}, clear=False), \
                 patch.object(app.shutil, "which", side_effect=lambda name: "/tmp/" + name if name in {"kwriteconfig6", "busctl", "kscreen-doctor"} else None), \
                 patch.object(app.subprocess, "run", side_effect=self._kwin_run(calls)), patch.object(app, "_screen_position", return_value="1560,0"):
                self.assertEqual(app.install_rule(), 0)
                self.assertEqual(app.install_rule(), 0)
                parser = app._read_config(config)
                ids = app._rule_ids(parser)
                self.assertEqual(ids, ["pip", "ShuraMonitorCompact"])
                self.assertEqual(parser.get("ShuraMonitorCompact", "title"), app.TITLE)
                self.assertEqual(parser.get("ShuraMonitorCompact", "titlematch"), "1")
                self.assertEqual(parser.get("ShuraMonitorCompact", "above"), "true")
                self.assertEqual(parser.get("ShuraMonitorCompact", "aboverule"), "2")
                self.assertEqual(parser.get("ShuraMonitorCompact", "position"), "1560,0")
                self.assertEqual(parser.get("ShuraMonitorCompact", "positionrule"), "3")
                self.assertEqual(len(list(root.glob("kwinrulesrc.bak-*"))), 2)
                self.assertEqual(app.remove_rule(), 0)
                parser = app._read_config(config)
                self.assertEqual(app._rule_ids(parser), ["pip"])
                self.assertEqual(parser.get("pip", "Description"), "Picture in Picture")
                self.assertFalse(parser.has_option("ShuraMonitorCompact", "title"))
                self.assertEqual(len([c for c in calls if Path(c[0]).name == "busctl"]), 3)

    def test_install_on_clean_file_creates_only_owned_rule(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            calls = []
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": temp, "XDG_CURRENT_DESKTOP": "KDE Plasma", "XDG_SESSION_TYPE": "x11"}, clear=False), \
                 patch.object(app.shutil, "which", side_effect=lambda name: "/tmp/" + name if name in {"kwriteconfig6", "busctl"} else None), \
                 patch.object(app.subprocess, "run", side_effect=self._kwin_run(calls)), patch.object(app, "_screen_position", return_value="0,0"):
                app.install_rule()
            parser = app._read_config(Path(temp) / "kwinrulesrc")
            self.assertEqual(app._rule_ids(parser), ["ShuraMonitorCompact"])
            self.assertEqual(parser.getint("General", "count"), 1)

    def test_missing_plasma_tools_does_not_write(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp, \
             patch.dict(os.environ, {"XDG_CONFIG_HOME": temp, "XDG_CURRENT_DESKTOP": "GNOME", "XDG_SESSION_TYPE": "wayland"}, clear=False), \
             patch.object(app.shutil, "which", return_value=None), patch("builtins.print") as output:
            self.assertEqual(app.install_rule(), 0)
            self.assertEqual(app.remove_rule(), 0)
            self.assertFalse((Path(temp) / "kwinrulesrc").exists())
            self.assertEqual(output.call_count, 2)

    def test_autostart_is_opt_in_and_removes_only_its_file(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp, patch.dict(os.environ, {"XDG_CONFIG_HOME": temp}, clear=False):
            path = Path(temp) / "autostart" / "shura-monitor.desktop"
            self.assertFalse(path.exists())
            self.assertEqual(app.autostart("on"), 0)
            exec_line = next(l for l in path.read_text().splitlines() if l.startswith("Exec="))
            self.assertTrue(exec_line.endswith(" monitor"), exec_line)
            self.assertIn("X-Shura-Managed=true", path.read_text())
            self.assertEqual(app.autostart("off"), 0)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
