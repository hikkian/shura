# Session cache on NVMe (SESS2): what it does, its switches, and the recommended values

Off by default (`sessionCacheEnabled` false). With the defaults the gateway behaves as before.

## What it keeps

- One snapshot per session, in the catalogue folder (`sessionCacheDir`, default `~/.local/state/ai-gateway/sessions`): a `.bin`
  file and a small `.json` record (session id, model key, tokens, times, prefix hash, size, and the gateway's own slot record).
- The session id comes from the `X-Shura-Session` header (a random uuid; the gateway accepts only that form). Without the header
  nothing is cached for the request.

## When it writes (write-behind, never during work)

Every condition must hold (`session_cache.should_write`):
- the model has been idle at least `sessionCacheIdleSeconds` (120);
- at least `sessionCacheMinGrowthTokens` (4000) new tokens since the last snapshot;
- the slot holds at least `sessionCacheMinSlotTokens` (8000) tokens;
- the disk I/O pressure (PSI `some avg10`) is below `sessionCacheIoPressureMax` (10); an unknown value counts as a refusal;
- no game or fullscreen video (the existing signals), and the RAM valve is not pressing or critical;
- the daily write budget (`ssdWriteDailyLimitGB`, shared with RAM1) allows it, and the catalogue quota and disk reserve allow it.

The slot is saved to the SSD slot file by the ordinary save, then copied to the catalogue with `ionice -c3`. The copy is fsynced and
renamed; the previous snapshot of the session is then replaced.

## When it reads (restore)

On a model load for a session that has a catalogue snapshot with the same compatibility key, the snapshot is placed where the
existing disk restore reads it, if restoring costs less than `sessionCacheRestoreRatio` (0.25) of recomputing the prefill at
`sessionCachePrefillTps`. Otherwise the prompt is re-read.

## Warm route

`POST /guardian/warm` (with the session header) loads the model and restores the session only when the valves allow it: not while a
game runs, not during an audio or video job, not while the RAM valve is above ok, not while the desktop needs VRAM. Otherwise it
answers `{"warm": false, "reason": ...}` and does nothing. Off unless `sessionCacheEnabled`.

## Limits (the disk never fills)

- Quota: the smaller of `sessionCacheQuotaGB` (30) and `sessionCacheQuotaPct` (8) of the partition.
- Reserve: no write leaves less than `sessionCacheReserveGB` (100) free; a full partition never takes a write.
- LRU to the quota, TTL `sessionCacheTTLDays` (14), and temporary files removed at every prune.
- Daily budget: `ssdWriteDailyLimitGB` (recommended 8, see the deviation note in the report).

## Recommended values (not applied to the live config)

- 31 GB machine: `sessionCacheEnabled` true; the defaults above; `ssdWriteDailyLimitGB` 8.
- Measure `sessionCacheReadGBps` and `sessionCachePrefillTps` on the machine before relying on the restore decision.

## Not wired (honest list)

- The session header on chat requests: ShuraCode does not send it yet (needs a provider-level header). The gateway accepts it.
- The ShuraCode plugin calls the warm route at start only when `shuracode.session_warm` is set in its kv (default off).
- The server's own write of the slot file is not in the idle I/O class (it happens inside llama-server); only the copy to the catalogue is.
- The tail-only prefill is left to the server's slot restore; no separate tail computation.
- Compaction-triggered prefill and the context-checkpoint schedule (design items 2 and 4) are not done.
