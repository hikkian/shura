# shura-desk-signal: a read-only KWin script for Shura

Tells Shura's gateway **which application is active and whether a window is full screen** (a game, a full-screen video), so it can give
the VRAM back before the desk asks for it. KWin on Wayland offers no read-only D-Bus call for this, so it takes a small script.

What it does: whenever the active window or the full-screen state changes it prints one line to KWin's log (the user journal):

```
shura-desk-signal {"v":1,"active":"brave-browser","fullscreen":true,"any_fullscreen":true,"fullscreen_class":"brave-browser"}
```

What it does **not** do: it does not move, resize, raise, minimize or close any window, does not set a property or a setting, does not
talk to the network or any service, and never prints window titles (only the application class). The code is 80 lines in
`contents/code/main.js`; `tests/test_kwin_script.py` runs it against a fake `workspace` whose windows throw on any write.
KWin's script API has no keyboard or mouse idle time, so nothing about input is reported; a locked screen is still seen through logind.

The gateway reads the newest line with `journalctl --user -b -t kwin_wayland -r -n 1 -g 'shura-desk-signal '` and only when this script is
enabled in `kwinrc`; without it the gateway keeps using the GPU-load proxy (`deskKwinSignal` in guardian.json switches the reading off).
The script adds one journal line per focus or full-screen change (a few hundred bytes; a few hundred changes a day).

## Install (two commands)

```bash
kpackagetool6 --type=KWin/Script --install ~/Projects/shura-integration/kwin/shura-desk-signal
kwriteconfig6 --file kwinrc --group Plugins --key shura-desk-signalEnabled true && busctl --user call org.kde.KWin /KWin org.kde.KWin reconfigure
```

Check that it works (focus another window, then):

```bash
journalctl --user -b -t kwin_wayland -r -n 3 -g 'shura-desk-signal ' -o cat
```

If no line appears, enable "Shura desk signal (read-only)" under System Settings → Window Management → KWin Scripts, or log out and in
once. (`~/.local/share/kwin/scripts/brave-pip-above` is another script of this kind already on this machine.)

## Remove

```bash
kwriteconfig6 --file kwinrc --group Plugins --key shura-desk-signalEnabled false && busctl --user call org.kde.KWin /KWin org.kde.KWin reconfigure && kpackagetool6 --type=KWin/Script --remove shura-desk-signal
```

## What was and was not tested

Tested: the script under node against a fake KWin (changes only, no titles, no writes, errors become a log line), the gateway's parser and
reader on canned journal output. **Not tested on a real KWin** (nothing was installed on this machine while writing it): the property names
(`workspace.activeWindow`, `workspace.windowList()`, `window.fullScreen`, `resourceClass`, `fullScreenChanged`, `minimizedChanged`,
`windowAdded`, `windowRemoved`, `windowActivated`) are the Plasma 6 ones, the same style as the script already in use here, but the first run
on the real desktop is the real test: the `journalctl` line above must show the active application and `fullscreen` flipping with F11.
