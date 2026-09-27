# Troubleshooting

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
Three consecutive failed loads. Read `journalctl --user -u ai-gateway -n 100`, fix the cause, then
`curl -X POST localhost:8080/guardian/ai-on`.

## Web search tool fails
`mcp-searxng` needs the JSON output format, which `searxng/settings.yml` enables. Test with
`curl "http://127.0.0.1:8888/search?q=test&format=json"`.

## `--prio 2` has no effect
Raising process priority on Linux requires root/`CAP_SYS_NICE`. The gateway intentionally does not
pass it.

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
