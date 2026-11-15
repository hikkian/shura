"""MEMV2: the read-only readers of the desk's signals, fed canned command output and a fake /proc tree. Nothing here touches the real
desktop, PipeWire or the GPU."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import desk_need
import desk_sources
from desk_sources import DeskSources, kwin_script_enabled, parse_fdinfo, parse_gamemode, parse_kwin_line, parse_pipewire, parse_session


def node(klass, state="running", **props):
    return {"type": "PipeWire:Interface:Node", "info": {"state": state, "props": {"media.class": klass, **props}}}


def fdinfo(client, vram_kib=1024, gfx=0, enc=0, driver="amdgpu"):
    lines = [f"drm-driver:\t{driver}", f"drm-client-id:\t{client}", f"drm-memory-vram:\t{vram_kib} KiB", f"drm-engine-gfx:\t{gfx} ns"]
    if enc:
        lines.append(f"drm-engine-enc:\t{enc} ns")
    return "\n".join(lines) + "\n"


class Parsers(unittest.TestCase):
    def test_gamemode(self):
        self.assertEqual(parse_gamemode(0, "(<uint32 2>,)\n"), 2)
        self.assertEqual(parse_gamemode(0, "(<0>,)"), 0)
        with self.assertRaises(ValueError):
            parse_gamemode(1, "")
        with self.assertRaises(ValueError):
            parse_gamemode(0, "garbage")

    def test_pipewire_call_and_playback(self):
        doc = json.dumps([node("Stream/Output/Audio"), node("Stream/Input/Audio"), node("Audio/Sink")])
        self.assertEqual(parse_pipewire(doc), (True, True))
        self.assertEqual(parse_pipewire(json.dumps([node("Stream/Output/Audio")])), (False, True))
        self.assertEqual(parse_pipewire(json.dumps([node("Stream/Input/Audio", **{"stream.capture.sink": True})])), (False, False))
        self.assertEqual(parse_pipewire(json.dumps([node("Stream/Input/Audio", state="idle")])), (False, False))
        self.assertEqual(parse_pipewire(json.dumps([node("Video/Source")])), (True, False))
        self.assertEqual(parse_pipewire(json.dumps([{"type": "PipeWire:Interface:Client"}, 5])), (False, False))
        with self.assertRaises(ValueError):
            parse_pipewire("not json")

    def test_session(self):
        self.assertEqual(parse_session("IdleHint=no\nIdleSinceHintMonotonic=0\nLockedHint=yes\n"), (None, True))
        idle, locked = parse_session("IdleHint=yes\nIdleSinceHintMonotonic=1\nLockedHint=no\n")
        self.assertFalse(locked)
        self.assertGreater(idle, 0)
        self.assertEqual(parse_session(""), (None, False))

    def test_fdinfo(self):
        info = parse_fdinfo(fdinfo(7, vram_kib=2048, gfx=5, enc=9))
        self.assertEqual((info["client"], info["vram_kib"], info["engines"]), ("7", 2048, {"gfx": 5, "enc": 9}))
        self.assertIsNone(parse_fdinfo(fdinfo(7, driver="i915")))
        self.assertIsNone(parse_fdinfo("pos:\t0\nflags:\t02\n"))


def kwin_line(**kw):
    data = {"v": 1, "active": "brave-browser", "fullscreen": False, "any_fullscreen": False, "fullscreen_class": ""}
    data.update(kw)
    return "shura-desk-signal " + json.dumps(data) + "\n"


class KwinParsers(unittest.TestCase):
    def test_line(self):
        got = parse_kwin_line(kwin_line(fullscreen=True, fullscreen_class="steam_app_1"))
        self.assertEqual((got["fullscreen"], got["fullscreen_class"], got["active"]), (True, "steam_app_1", "brave-browser"))
        # a journal prefix in front of the tag does not matter, the newest tagged line wins
        text = "Oct 25 kwin_wayland[1]: " + kwin_line(active="old") + "noise\n" + "kwin_wayland: " + kwin_line(active="new")
        self.assertEqual(parse_kwin_line(text)["active"], "new")

    def test_bad_lines(self):
        for bad in ("", "nothing here\n", "shura-desk-signal {broken\n", 'shura-desk-signal {"v":2}\n',
                    'shura-desk-signal {"v":1,"error":"x"}\n', "shura-desk-signal [1]\n"):
            with self.assertRaises(ValueError, msg=bad):
                parse_kwin_line(bad)

    def test_enabled_in_kwinrc(self):
        self.assertTrue(kwin_script_enabled("[Plugins]\nfoo=1\nshura-desk-signalEnabled=true\n"))
        self.assertFalse(kwin_script_enabled("[Plugins]\nshura-desk-signalEnabled=false\n"))
        self.assertFalse(kwin_script_enabled("[Other]\nshura-desk-signalEnabled=true\n"))
        self.assertFalse(kwin_script_enabled(""))


class FakeProc:
    """A /proc tree in a temp folder: pid -> comm and fdinfo files."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def add(self, pid, comm, files):
        d = self.root / str(pid)
        (d / "fdinfo").mkdir(parents=True, exist_ok=True)
        (d / "comm").write_text(comm + "\n")
        for name, text in files.items():
            (d / "fdinfo" / str(name)).write_text(text)

    def cleanup(self):
        self.tmp.cleanup()


class Collect(unittest.TestCase):
    def setUp(self):
        self.proc = FakeProc()
        self.addCleanup(self.proc.cleanup)
        self.now = 100.0
        self.outputs = {"gdbus": (0, "(<uint32 0>,)\n"), "pw-dump": (0, "[]"), "loginctl": (0, "IdleHint=no\nIdleSinceHintMonotonic=0\nLockedHint=no\n")}
        self.calls = []

    def sources(self):
        def run(cmd, timeout=2.0):
            self.calls.append(cmd)
            return self.outputs[cmd[0]]
        return DeskSources(dict(desk_need.DEFAULTS), run=run, proc_root=str(self.proc.root), clock=lambda: self.now, uid=os.getuid())

    def test_quiet_desk(self):
        self.proc.add(10, "plasmashell", {"5": fdinfo(1)})
        sig = self.sources().collect()
        self.assertEqual((sig.game, sig.gpu_app, sig.call, sig.video, sig.locked, sig.errors), (False, "", False, False, False, []))

    def test_gamemoded_client_is_a_game(self):
        self.outputs["gdbus"] = (0, "(<uint32 1>,)\n")
        self.assertTrue(self.sources().collect().game)

    def test_a_foreign_process_that_keeps_the_graphics_engine_busy_is_a_game(self):
        self.proc.add(20, "somegame", {"4": fdinfo(9, vram_kib=500 * 1024, gfx=0)})
        s = self.sources()
        s.collect()
        self.now += 2.0
        self.proc.add(20, "somegame", {"4": fdinfo(9, vram_kib=500 * 1024, gfx=1_800_000_000)})   # 90 % of 2 s
        sig = s.collect()
        self.assertEqual(sig.gpu_app, "somegame")
        self.assertIn("gfx 90%", sig.gpu_app_detail)

    def test_a_foreign_process_with_much_vram_is_a_game_even_when_idle(self):
        self.proc.add(21, "bigapp", {"4": fdinfo(9, vram_kib=3000 * 1024)})
        self.assertEqual(self.sources().collect().gpu_app, "bigapp")

    def test_desktop_and_shura_are_never_games(self):
        self.proc.add(30, "llama-server", {"4": fdinfo(1, vram_kib=12000 * 1024)})
        self.proc.add(31, "kwin_wayland", {"4": fdinfo(2, vram_kib=4000 * 1024)})
        self.proc.add(32, "brave", {"4": fdinfo(3, vram_kib=300 * 1024)})
        self.assertEqual(self.sources().collect().gpu_app, "")

    def test_the_model_pid_is_excluded(self):
        self.proc.add(40, "unknownthing", {"4": fdinfo(1, vram_kib=9000 * 1024)})
        self.assertEqual(self.sources().collect(model_pids=(40,)).gpu_app, "")

    def test_several_fds_of_one_client_count_once(self):
        self.proc.add(50, "brave", {"4": fdinfo(8, vram_kib=1500 * 1024), "5": fdinfo(8, vram_kib=1500 * 1024)})
        self.assertEqual(self.sources().collect().browser_vram_mib, 1500)

    def test_video_is_playback_sound_with_the_video_engine_busy(self):
        self.outputs["pw-dump"] = (0, json.dumps([node("Stream/Output/Audio")]))
        self.proc.add(60, "brave", {"4": fdinfo(8, enc=0)})
        s = self.sources()
        s.collect()
        self.now += 2.0
        self.proc.add(60, "brave", {"4": fdinfo(8, enc=400_000_000)})       # 20 % of the video engine
        self.assertTrue(s.collect().video)

    def test_music_alone_is_not_video(self):
        self.outputs["pw-dump"] = (0, json.dumps([node("Stream/Output/Audio")]))
        self.proc.add(60, "brave", {"4": fdinfo(8)})
        s = self.sources()
        s.collect()
        self.now += 2.0
        self.assertFalse(s.collect().video)

    def test_call_and_lock(self):
        self.outputs["pw-dump"] = (0, json.dumps([node("Stream/Input/Audio")]))
        self.outputs["loginctl"] = (0, "IdleHint=no\nIdleSinceHintMonotonic=0\nLockedHint=yes\n")
        sig = self.sources().collect()
        self.assertTrue(sig.call)
        self.assertTrue(sig.locked)

    def test_a_failing_reader_names_the_failure_and_stays_neutral(self):
        self.outputs["pw-dump"] = (-1, "")
        self.outputs["loginctl"] = (-1, "")
        self.outputs["gdbus"] = (0, "garbage")
        sig = self.sources().collect()
        self.assertEqual((sig.call, sig.locked, sig.idle_s, sig.game), (False, False, None, False))
        self.assertEqual(len(sig.errors), 3)
        # unknown idle is ordinary use, never "away"
        self.assertEqual(desk_need.target_need(sig, dict(desk_need.DEFAULTS))[0], 0.3)

    def test_gamemoded_not_running_is_not_an_error(self):
        self.outputs["gdbus"] = (1, "")
        self.assertEqual(self.sources().collect().errors, [])

    def test_the_fd_list_is_looked_up_once_not_every_poll(self):
        self.proc.add(70, "brave", {"4": fdinfo(8)})
        s = self.sources()
        s.collect()
        (self.proc.root / "70" / "fdinfo" / "6").write_text(fdinfo(9, vram_kib=7 * 1024))   # a new fd appears
        self.now += 2.0
        self.assertEqual(s.collect().browser_vram_mib, 1)                                   # not seen yet (cached list)
        self.now += 31.0
        self.assertEqual(s.collect().browser_vram_mib, 1 + 7)                               # after the rescan interval

    def kwin_sources(self, enabled=True, line=None, rc=0):
        kwinrc = self.proc.root / "kwinrc"
        kwinrc.write_text("[Plugins]\nshura-desk-signalEnabled=true\n" if enabled else "[Plugins]\n")
        self.outputs["journalctl"] = (rc, line if line is not None else kwin_line())
        s = self.sources()
        s.kwinrc = kwinrc
        return s

    def test_kwin_script_off_means_no_journal_search_and_the_proxy(self):
        s = self.kwin_sources(enabled=False)
        sig = s.collect()
        self.assertIsNone(sig.fullscreen)
        self.assertFalse(any(c[0] == "journalctl" for c in self.calls))

    def test_kwin_fullscreen_game(self):
        s = self.kwin_sources(line=kwin_line(active="steam_app_620", fullscreen=True, any_fullscreen=True, fullscreen_class="steam_app_620"))
        sig = s.collect()
        self.assertTrue(sig.fullscreen)
        self.assertEqual(sig.active_class, "steam_app_620")
        self.assertEqual(desk_need.target_need(sig, dict(desk_need.DEFAULTS))[0], 1.0)

    def test_kwin_fullscreen_browser_is_video_level_and_work_tools_are_ignored(self):
        cfg = dict(desk_need.DEFAULTS)
        browser = self.kwin_sources(line=kwin_line(fullscreen=True, any_fullscreen=True, fullscreen_class="brave-browser")).collect()
        self.assertEqual(desk_need.target_need(browser, cfg)[0], 0.6)
        for work in ("org.kde.konsole", "code", "com.mitchellh.ghostty", "telegram-desktop"):
            sig = self.kwin_sources(line=kwin_line(active=work, fullscreen=True, any_fullscreen=True, fullscreen_class=work)).collect()
            self.assertEqual(desk_need.target_need(sig, cfg)[0], 0.3, work)

    def test_kwin_fullscreen_on_another_screen_counts(self):
        sig = self.kwin_sources(line=kwin_line(active="code", fullscreen=False, any_fullscreen=True, fullscreen_class="vlc")).collect()
        self.assertEqual(desk_need.target_need(sig, dict(desk_need.DEFAULTS))[0], 1.0)

    def test_kwin_line_cached_between_polls_and_a_failed_search_is_not_repeated(self):
        s = self.kwin_sources()
        s.collect()
        s.collect()
        self.assertEqual(len([c for c in self.calls if c[0] == "journalctl"]), 1)         # within kwin_poll_seconds
        bad = self.kwin_sources(rc=1)
        sig = bad.collect()
        self.assertIsNone(sig.fullscreen)
        self.assertTrue(any("kwin script" in e for e in sig.errors))
        self.now += 10.0
        bad.collect()
        self.assertEqual(len([c for c in self.calls if c[0] == "journalctl"]), 2)         # no new search for the retry interval
        self.now += 400.0
        bad.collect()
        self.assertEqual(len([c for c in self.calls if c[0] == "journalctl"]), 3)

    def test_kwin_can_be_switched_off_by_config(self):
        s = self.kwin_sources(line=kwin_line(fullscreen=True, fullscreen_class="game"))
        s.cfg["deskKwinSignal"] = False
        self.assertIsNone(s.collect().fullscreen)

    def test_foreign_users_processes_are_not_read(self):
        self.proc.add(80, "somegame", {"4": fdinfo(9, vram_kib=5000 * 1024)})
        s = DeskSources(dict(desk_need.DEFAULTS), run=lambda cmd, timeout=2.0: self.outputs[cmd[0]], proc_root=str(self.proc.root),
                        clock=lambda: self.now, uid=os.getuid() + 1)
        self.assertEqual(s.collect().gpu_app, "")


if __name__ == "__main__":
    unittest.main()
