# Operations

## Public benchmarks

The reproducible external harness, task preparation, metrics, and safety rules are documented in
[bench/agentic-eval/external/README.md](../bench/agentic-eval/external/README.md). These public harnesses
are separate from private model files and do not require a live gateway for their unit tests.

## Monitor

The gateway serves the read-only dashboard at `/monitor` and its JSON/SSE endpoints under
`/monitor/api/`. The page displays operational telemetry and bounded request metadata, but does not
return prompt or response text. `shura monitor` opens the compact desktop window, while
`shura monitor --full` opens the regular page. The compact view is also available as
`/monitor?view=compact`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the data fields, authentication, and
polling behavior.

## Guardian options

`config/guardian.example.json` is the shipped reference. `crashRecovery` defaults to `false` and requires
RAM-backed saved-slot checkpoints (`slotSaveCheckpoints >= 1`). When enabled, the gateway may retry a
request only after an unexpected backend exit and only when it can prove the previous response was not
already delivered; partial/uncertain responses are not replayed. It also depends on enough available RAM
and the configured `/dev/shm` slot directory. Keep it disabled until the host-specific recovery checks
in `ARCHITECTURE.md` pass. Other default-off options such as `vramGuard`, `slotGears`, `politeModel`, and
`gameMode` are independent and have their own prerequisites.

Memory-pressure load admission is opt-in and its keys belong in `guardian.json`, not `model-launch.json`.
It runs only in the new gateway path; `vramGuard: true`, `slotSaveCheckpoints > 0`, or another new-gateway
feature selects that path. If an enabled or malformed pressure key is present while the legacy path is
selected, the gateway logs that the option is inactive. The example starts with both thresholds at 0
and suggests `zramMaxPctToLoad: 60` and `psiMemoryFullMaxToLoad: 5` as starting points.

`zramMaxPctToLoad` checks logical fill: uncompressed zram data divided by device capacity. The refusal
also reports the actual RAM used by zram in MiB. A given fill percentage can represent different RAM
amounts depending on how compressible the stored pages are. On systems with no zram device (including
zswap-only systems), this key cannot protect memory. `psiMemoryFullMaxToLoad` checks `full avg10` from
memory PSI. `full` is time when all non-idle tasks are stalled on memory; 5% avg10 means about half a
second of such stalls in the last ten seconds. Current zram and PSI readings are on the monitor page;
they can also be read with `cat /proc/pressure/memory`.

For either threshold, omit the key or set it to `0`, `null`, `false`, or a negative number to disable
that check. Positive values must be finite percentages no greater than 100. Invalid values on the new
gateway stop its startup with the key named; invalid values ignored by the legacy path are warned about
in the journal. A telemetry read/parse error logs once for the unavailable period and falls back to the
existing RAM/VRAM checks. When pressure is too high, the gateway returns HTTP 503 with the measured
value, threshold, and `Retry-After` (30 seconds by default; tune with
`memoryAdmissionRetryAfterSeconds`). Wait for the pressure to ease, check the readings, then retry; the
gateway does not wait in a loop. The retry setting is an integer from 1 to 3600 seconds. `shura warm`
is an ordinary generation request and follows the same pressure gate. Safety restarts for a slot-gear
rollback or an eligible pre-response crash replay skip only the new pressure check; their existing RAM
and VRAM admission checks still apply.

### A failed switch to the long context

When the switch from the short to the long layout fails for a reason that is not memory (the slot checkpoint before the switch, the long
server's start), the request gets HTTP 503 with the error code `SHURA_LAYOUT_SWITCH_FAILED`, the reason, and a `Retry-After`; the short layout
is loaded again and keeps serving. The same switch is not tried again for `layoutSwitchRetrySeconds` (60 by default, an integer from 1 to
3600): a client that retries gets the same answer at once instead of making the gateway stop and start the server on every retry. A memory
refusal keeps its own code and `Retry-After` and may be tried again at the next request. `GET /guardian/status` shows
`layout_switch.state` (`failed`) and `layout_switch.error`. A server that does not report `n_slot_save_estimated_bytes` in `/slots` (shura.cpp
before commit c4d92d5e4) cannot checkpoint a session and fails every switch this way.

The optional vramCardReserveMiB rule selects the starting expert-cache level from fresh NVML free
VRAM for the whole card. Missing or zero keeps the configured slot count; 194 MiB is the recommended
reserve for the tested desktop. The estimate uses the model's existing vramFreeMinGBToLoad as its
non-cache load baseline and adds slotGearMiBPerSlot per cache slot, measured at about 36.156 MiB.
The selected level is the largest supported slotGearLevels value no greater than the model's
configured moeCacheSlots, with 16 as the minimum supported gear. A model configured below 16 keeps
its fixed level. If even the minimum level cannot leave the reserve, the gateway defers startup
without launching llama-server. This initial choice does not replace the independent slotGears
controller; later slot changes keep using its existing ratchet. After selection, shura status shows
the model, selected slots, and estimate in the existing slot gears line; guardian/status includes
the same decision in vram_card_reserve.

This is an admission estimate, not a promise against later desktop or driver memory changes. Keep
the runtime VRAM safety checks enabled as appropriate and verify actual free-card VRAM after
deployment. A malformed nonzero reserve value stops the new gateway at startup with a clear config
error; null, false, and zero disable this rule.

The separate vramValve option is experimental and defaults off. It requires vramGuard=true,
slotSaveCheckpoints>=1, a positive vramCardReserveMiB, and a llama-server built with live resize;
it cannot run alongside slotGears. On fresh free-card VRAM below the reserve, it calls the server's
in-place resize endpoint only at an idle request boundary. The server independently rejects resize
if any slot is active. It chooses the highest lower supported level that should restore the
reserve. The default ladder is 18/16/8; zero is excluded because the GLIVE token check found a
repeatable output mismatch there. If the configured minimum positive level still cannot restore the
reserve for two seconds, it saves A2 and uses the existing vramGuard parking path. It restores one
level at a time only after 60 seconds above reserve plus the next level's estimated slot cost plus
200 MiB hysteresis. Missing/stale telemetry never resizes. A model entry may override the shared
`vramValveLevels` with its own descending list; levels above that model's `moeCacheSlots` are clipped.
It may also set `moeCacheSlotMiB`. The price priority is valid per-model config, then the in-memory
mean of the last three uncontended settled resize measurements for that model, then
`slotGearMiBPerSlot`; measurements are never persisted. Values outside 10–80 MiB/slot are rejected.
The example's [18,16,8] ladder is sized for an 18-slot IQ4_XS setup. GLIVE runs measured about
31.3 MiB/slot for Tiel APEX and 36.2 MiB/slot for IQ4_XS. For APEX configured at 54 or 60 slots,
use a per-model list such as [60,40], which clips to [54,40] or [60,40]. Occamy at 12 slots with a
shared [54,40] list clips to [12], so the valve has no lower step; no Occamy valve ladder has been
validated for token quality or recovery. Vision mode disables the valve and status reports the
vision slot count. Every transition or failure appears in the journal, guardian/status and shura
status. Keep this off until the live-map, token-equivalence, cycle and desktop-comfort gates in the
latest GLIVE report pass.

Known per-model examples (do not copy an IQ4_XS price onto APEX):

| Model family | `moeCacheSlotMiB` | Example valve levels | Evidence |
|---|---:|---|---|
| Tiel APEX-I-MiniPlus V2.1 | 31.3 | `[54,40]` or `[60,40]` | 625.3 MiB released over 20 slots in GLIVE5; GLIVE6's single 30.7 estimate was not accepted because its probe missed graphics-process changes |
| IQ4_XS | 36.2 | Set for that model's maximum | Prior static resize measurements / configured default |
| Occamy, 12 slots | Not measured in GLIVE6 | `[12]` under shared `[54,40]` | Inert; no Occamy quality or return validation |

Per-model overrides belong inside each `models.<id>` entry in the example model-launch file; the
live file is not edited by these experiments.
