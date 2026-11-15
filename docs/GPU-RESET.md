# GPU reset is not a model crash (GPURESET1)

A full reset of the graphics card (amdgpu: `ring gfx_0.0.0 timeout`, MES does not answer, `GPU reset begin`, `VRAM is lost`) kills every
process that holds VRAM. llama-server then dies with `ROCm error: unspecified launch failure` (exit -6) although the model did nothing
wrong. The first such event on this machine (2026-10-25 14:12, the desktop's own graphics ring hung first; see
`reports/REPORT-GPURESET1.md`) looked like a crash of the model.

What the gateway does now after any unexpected exit of the server (not the governor's 75):

* reads the kernel log since the server started (`journalctl -k`, a few ms, once per process) and looks for `GPU reset begin` /
  `VRAM is lost`. A ring timeout that the driver recovered by itself, or a page fault, is not a card reset;
* if it finds one: `last_error` and the status say `GPU reset (ring gfx_0.0.0 timeout in kwin_wayland; VRAM lost: yes) killed llama-server
  (exit -6); the model is not at fault`, the event `gpu_reset` is logged, and `GET /guardian/status` has `gpu_resets` (total, in the
  window, the last one's details);
* the reset is **not** a strike against the model's crash limit (`crashRestartMaxAttempts`). It counts in its own limit:
  `gpuResetMaxPerWindow` (3) resets inside `gpuResetWindowSeconds` (900) stop the restarts (ERROR state, like the crash limit), because a
  card that keeps resetting needs a person, not a loop;
* the server starts again with the same layout at the next request (as after any crash).

`gpuResetRetry` (default `false`): a request that had not produced a token yet when the reset killed the server is replayed once on the
restarted server, into the same open response (the same mechanism as `vramRetry`, but only for a GPU reset; with `vramRetry` on any stop
before the first token is replayed). After the first token nothing is replayed; the client gets the honest interruption error. The switch
puts POST requests on the recovery path (one protected generation at a time), so it is off by default.

Benchmarks and stands should also check `journalctl -k --since <start> | grep -E "GPU reset begin|VRAM is lost"` after a run and treat a
run with a reset as invalid.
