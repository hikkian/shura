# Troubleshooting

## First install

Before a real install, run `./scripts/shura check` (installs nothing) and `./scripts/shura install --dry-run` (downloads
nothing). They show the model, the quant, the disk space and the backends that will be tried. The installer checks the
free space before it downloads, and it resumes an interrupted download.

The messages below come from the installer itself, so you can search for them.

**`Install stopped: … is not in the … repository`**
The download source has no file for the planned model. This is a catalog problem, not a problem with your computer. Open an
issue with the install-failed template (`.github/ISSUE_TEMPLATE/install-failed.md`).

**`checksum mismatch for …`**
The downloaded file does not match its SHA-256, so the installer does not use it. Run `./scripts/shura install` again. If it
repeats, check for a proxy or an antivirus that rewrites downloads.

**`No GPU build works here: planning for the CPU instead.`**
No GPU build (Vulkan, for example) started on this machine. The plan falls back to the CPU: it works, but it is slower.
Check the GPU driver first. On Linux, `vulkaninfo --summary` should list your card.

**`The requested <backend> backend has no fitting plan here; planning the Vulkan fallback.`**
The backend you asked for with `--backend` cannot run the model in your memory. The installer uses Vulkan instead. Run
the install without `--backend`.

**`<engine> did not work here (…)`**
One engine failed its check: the server did not start with the planned settings. The installer tries the next one. Only
the final `Install stopped:` line means that nothing worked.

**Speed is far below the prediction**
The prediction before the first run is an estimate. The installer measures the real speed on your machine and corrects the
plan. The prediction for MoE models on Vulkan is not calibrated yet ([SPEED-ESTIMATES.md](SPEED-ESTIMATES.md)), so it can be
far from what you will see. The measurement is the number to trust.

Before you open an issue, run `./scripts/shura doctor` and paste its output, with your user name and paths removed.

## ShuraCode problems
Run `shuracode doctor`. Agent-side issues (UI, memory, commands, engine updates) are covered in the
[ShuraCode repository](https://github.com/hikkian/shuracode). Two engine behaviours ShuraCode already
handles: the engine can hang at startup when it cannot reach its own servers (ShuraCode runs it offline),
and the model sometimes mistyped the home directory in absolute paths (ShuraCode's rules make it use
project-relative paths).

## `GGML_ASSERT(n_kv_max > 0) failed` with `-ctk turbo3`
Unpatched TurboQuant on a head_dim-256 model. Rebuild with `scripts/build-llama.sh` (applies the fix),
then run `bench/correctness_probes.py`. See [TURBOQUANT-FIX.md](TURBOQUANT-FIX.md).

## CUDA out of memory mid-generation, although the model loaded fine
Too many `--moe-cache-slots`. Transient compute buffers at full context need more VRAM than a fresh
load. Reduce slots (24 with turbo3 / 16 with q4_0 on 12 GB) and re-test at full context.

## Decode gets slower with every request
Usually VRAM oversubscription from too many expert slots (it does not error, it degrades). Also
check that `GGML_CUDA_REGISTER_HOST` / `GGML_SCHED_PREFETCH_EXPERTS` are **not** set together with the
expert cache.

## A restored slot is ignored and the whole prompt is re-processed
Expected if you resend the *identical* prompt: the hybrid Gated-DeltaNet architecture cannot rewind its
recurrent state to re-evaluate the last token. Continuations of the conversation reuse the slot.

## Gateway returns 503 "Not enough free RAM to load the model"
`ramFreeMinGBToLoad` (default 14 GB) guards against load/unload thrashing on a 32 GB machine. Close
something or lower it in `config/guardian.json`. Check `curl -s localhost:8080/guardian/status`.

## Gateway is in `ERROR`
The gateway goes to `ERROR` after `crashRestartMaxAttempts` consecutive crashes or exits (3 in the example config): a server that exits while loading or after it was running. A load that only times out on the health check does not count. Read `journalctl --user -u ai-gateway -n 100`, fix the cause, then
`curl -X POST localhost:8080/guardian/ai-on`.

## Web search tool fails
`mcp-searxng` needs the JSON output format, which `searxng/settings.yml` enables. Test with
`curl "http://127.0.0.1:8888/search?q=test&format=json"`.

## `--prio 2` has no effect
Raising process priority on Linux requires root/`CAP_SYS_NICE`. The gateway intentionally does not
pass it.

## Model load deferred by zram or memory PSI
Wait for pressure to ease, check zram/PSI on `/monitor` or `cat /proc/pressure/memory`, then retry after `Retry-After`. The opt-in keys are in `guardian.json` and only work on the new gateway path.

## CUDA out of memory with `--spec-draft-type-k/v q8_0` (or `q4_0`)
The MTP draft context has no flash-attention kernel for quantized KV at head_dim 256. llama.cpp falls
back to dequantizing the full 200k-token draft cache into a temporary buffer, and VRAM runs out. Leave
the draft KV at its default (F16).

## The desktop stutters while the model generates
Use `threads` = physical cores (6 on a Ryzen 5 5600) and keep `niceLevel` at 10. Also check free VRAM:
browsers and IDEs render on the GPU too, so leave them a few hundred MB (see BENCHMARKS.md).

## GNOME Shell crashed right after running `install.sh`
GNOME watches `~/.local/share/applications` live. gnome-menus 3.38 crashes gnome-shell if it reads a
desktop entry while the file is still being written. Earlier versions of `install.sh` overwrote the
launcher in place and could trigger this. The installer now replaces files atomically and leaves them
untouched when nothing changed. Log back in; nothing else needs fixing.
