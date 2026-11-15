# Desk need: Shura reads the desk and gives way before it is asked (MEMV2)

With the VRAM governor on (`vramPolicy: "governor"`) and `deskNeed: true`, the gateway listens to the desk and tells the server's
governor how polite to be. **Off by default.** It only reads: it changes no setting of KDE, the session, PipeWire or the GPU.

## What it reads

| signal | source | how |
|---|---|---|
| game | gamemoded client count | `gdbus` on the session bus |
| heavy 3D / game without gamemoded | amdgpu fdinfo of the user's own processes | a foreign process keeping the graphics engine ≥ `deskGpuAppBusyPercent` (50 %) or holding ≥ `deskGpuAppVramMiB` (2 GiB); the desktop itself, browsers, editors and Shura's server are never games |
| call, camera, screen sharing | PipeWire graph | `pw-dump`: a running capture stream (microphone, camera, screen) |
| video | PipeWire + fdinfo | a running playback stream while a browser's video engine (or, with software decoding, its graphics engine) is busy; music alone is not video |
| browser VRAM | fdinfo | ≥ `deskBrowserVramMiB` (2 GiB) lifts the need to the video level |
| nobody there | `loginctl show-session` | the lock screen, or IdleHint when the session sets it |

KWin's active window and full-screen state on Wayland have no read-only D-Bus call; with the user's agreement a small read-only KWin
script, `kwin/shura-desk-signal/` (README there: two commands to install, one to remove), prints them to the user journal. When it is
enabled in `kwinrc` the gateway reads its newest line: a full-screen browser is the video level (0.6), a full-screen application that is
not a tool of the person (terminal, editor, file manager, messenger...) is a game (1.0), a full-screen window on another screen counts
too. Without the script the gateway uses the GPU-load proxy above (`deskKwinSignal: false` switches the reading off).
The input idle time on KDE cannot be read (logind's IdleHint stays `no`, KWin's script API has no input time), so a locked screen is the
only "nobody is there" signal; an unlocked idle desk counts as ordinary use.
A signal that cannot be read is named in `desk_need.signals.errors` and never lowers the need.

## From signals to the number

`human_need` ∈ {0, 0.3, 0.6, 1.0}: 0 locked / long idle, 0.3 ordinary use, 0.6 video, call or a heavy browser, 1.0 a game.
It **rises at once** and **falls only after `deskNeedDownSeconds` (60 s)** of being lower (to the highest value seen in that time), so a
pause in a video does not give the cache back.

The reserve of free VRAM: `0.75 GiB + 1.75 GiB × need`, rounded up to 64 MiB: 0.75 at night … 1.5 ordinary … 2.5 in a game, and
**never below `deskReserveFloorMiB` (1536 by default)**: the 1.5 GiB floor of the VRAM watchdog stands until the user decides otherwise.
With the default floor, need 0 and 0.3 both give 1536; 0.6 gives 1856; 1.0 gives 2560.

Light sleep (`sleep_level: 1`): the desk itself asked for 1.0 for `deskSleepOnSeconds` (5 s) in a row. The server releases the whole
expert cache and keeps the model in VRAM. It wakes when the (held) need has been below 1.0 for `deskSleepOffSeconds` (60 s); the cache
then grows back through the governor's usual calm period.

Deep sleep (L2, `deskDeepSleep`, on by default): the model leaves the card. It starts when the desk has asked for 1.0 for
`deskDeepSleepSeconds` (120 s) in a row, or at once when the governor had to pause the server during light sleep (the game needs more
than the cache could give back; without deep sleep the gateway would bring the model straight back into the game's way). It parks the
model through game mode's machinery: a running answer is drained, the context is saved to RAM (not to the SSD), the server stops, and
requests get a short "the model is paused because of a game" with Retry-After. It ends together with light sleep; the model then loads
on the next request. Game mode's own detectors can stay off: with `deskNeed` and the governor, the parking is on.

## What it tells the server

`POST /vram/policy` with `reserve_mib`, `sleep_level`, `human_need` (informational) and `note` (the reason, e.g. `gpu_app: foo gfx 80%`),
only when the reserve moved by ≥ `deskReserveStepMiB` (128) or the sleep level changed, and not more often than every
`deskPushMinSeconds` (5). A restarted server gets everything again. The server keeps a journal of what it gave back
(`GET /vram/journal`, in memory); the gateway reads it every 5 s and keeps daily totals in `~/.local/state/ai-gateway/desk-ledger.json`
(written at most every 15 minutes and only when it changed: well under a megabyte a day; the last 30 days are kept).

## What it shows

`GET /guardian/status` → `desk_need`: `enabled`, `need`, `reason`, `reserve_mib`, `floor_mib`, `sleep_level`, `sleeping_s`, `today`
(`yields`, `yielded_mib`, `recovered`, `recovered_max_s`, `pauses`, `sleeps`, `sleeping_s`), `signals`, `status_line`, `error`.
`status_line` is one string for a status bar, e.g. `Шура уступила 3× за сегодня (1.0 ГиБ) · карта освобождалась ≤ 0.31 с · нужда стола 0.6`.

## Settings (guardian.json)

| key | default | meaning |
|---|---|---|
| `deskNeed` | `false` | switch (needs `vramPolicy: "governor"`) |
| `deskReserveFloorMiB` | 1536 | lowest reserve ever pushed (256..4096); lowering it is the user's decision |
| `deskNeedDownSeconds` | 60 | hold before the need falls |
| `deskIdleSeconds` | 600 | idle time that counts as nobody there (only when the session reports it) |
| `deskGpuAppBusyPercent`, `deskGpuAppVramMiB`, `deskBrowserVramMiB` | 50, 2048, 2048 | thresholds above |
| `deskSleepOnSeconds`, `deskSleepOffSeconds` | 5, 60 | light sleep |
| `deskDeepSleep`, `deskDeepSleepSeconds` | `true`, 120 | deep sleep: park after this long in a game, or at once on a pause in light sleep (0 s: only on a pause) |
| `deskReserveStepMiB`, `deskPushMinSeconds` | 128, 5 | push rate limits |
| `deskPollSeconds` | 2 | how often the desk is read |
| `deskKwinSignal` | `true` | read the KWin script's line when the script is enabled |
| `deskSignalsFile` | `""` | test hook: read the signals from this JSON file instead of the desktop |

## Cost

One poll reads a few small files and runs `gdbus`, `pw-dump` and `loginctl` (about 10–30 ms of CPU); the list of graphics file
descriptors of the user's processes is rebuilt every 30 s (about 0.2 s of CPU), so the average is below 1 % of a core. It runs in its
own thread, apart from the VRAM guard.

## Tests

`tests/test_desk_need.py` (logic), `test_desk_sources.py` (readers, canned output and a fake /proc), `test_desk_runtime.py` (against the fake
governor on a virtual clock), `test_gateway_desk_need.py` (the real gateway against the fake llama-server). Nothing touches the real
desktop or the GPU.
