# RAM valve (RAM1): what it does, its switches, and the recommended values

The valve is opt-in. Every switch below is off by default, and with the defaults the gateway behaves as before.

## Switches (guardian.json)

| key | default | what it does |
|---|---|---|
| `llamaOomScoreAdj` | `0` | After the model process starts, write this value to `/proc/<pid>/oom_score_adj` (range -1000..1000; `800` makes the kernel kill the model before the desktop). A failed write is only logged (`llama_oom_score_adj_failed`). |
| `ramGuard` | `false` | Turn on the RAM level policy. It reads MemAvailable, PSI memory `full avg10` and zram fill on each memory poll. Level changes are recorded as the event `ram_guard_level`. **No action is taken yet**: see "Not wired". |
| `ramGuardPollSeconds` | `5` | Poll period (the gateway's existing memory poll is used; no new timer). |
| `ramGuardTightMemAvailableGB` / `ramGuardPressingMemAvailableGB` / `ramGuardCriticalMemAvailableGB` | `6.0` / `3.0` / `1.5` | MemAvailable thresholds for the levels tight / pressing / critical. |
| `ramGuardPsiFullTight` / `…Pressing` / `…Critical` | `5.0` / `15.0` / `40.0` | PSI memory `full avg10` thresholds (percent). |
| `ramGuardZramPctTight` / `…Pressing` / `…Critical` | `60` / `80` / `90` | zram fill thresholds (percent of the zram device). |
| `ramGuardHysteresisGB` | `0.5` | A level is left only when MemAvailable is this far above its entry threshold (PSI and zram use 1 and 2 percentage points). |
| `ramCacheBudgetPct` | `0` | Budget for what the model keeps in RAM beyond the weights (`--cache-ram` and `--ctx-checkpoints`), as a percent of MemTotal. `0` = no budget. The gateway lowers `--cache-ram` first, then the checkpoint count, until the sum fits. Recommended `10` (about 3 GB on 31 GB). |
| `ssdWriteDailyLimitGB` | `0` | Daily limit for ordinary slot saves to disk (state in `ssd-write-budget.json` in the slot folder). Over the limit a save is skipped and logged; the session is re-read later. `0` = no limit. Recommended `8` per day. |

## Levels

`ok` → `tight` → `pressing` → `critical`. A reading that crosses a threshold enters its level at once (the worst of the three readings
decides). Leaving a level needs the readings to recover past its threshold plus the hysteresis, and it steps down one level per poll.

## Recommended values (not applied to the live config)

- 31 GB machine (this one): `llamaOomScoreAdj` 800; `ramGuard` true with the default thresholds; `ramCacheBudgetPct` 10; `ssdWriteDailyLimitGB` 8.
- 16 GB machine: the same switches, with `ramGuardTightMemAvailableGB` 4.0, `ramGuardPressingMemAvailableGB` 2.0, `ramGuardCriticalMemAvailableGB` 1.0.

## Not wired (honest list)

- Actions on `pressing` (slow the new prefill, lower the CPU threads, drop cache) and on `critical` (unload with the slot kept). Only the level and its event exist.
- A llama-server API to drop the prompt cache at runtime was not checked; `--cache-ram` is set at start only.
- The live gateway (`guardian.json` with `slotSaveCheckpoints` 1) uses the current gateway, where the switches apply. The legacy gateway (`_gateway_legacy.py`, selected when none of `vramGuard`, `slotSaveCheckpoints`, `politeModel`, `slotGears`, `crashRecovery`, `vramCardReserveMiB`, `vramValve` is set) has no RAM valve.
