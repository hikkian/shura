# External quality benchmarks

This directory keeps the public-reference pipeline separate from the v1/v2 task
sets. Use a `/dev/shm` workspace for source clones, imported exercises, generated
tasks, sessions, caches, and results. The agent and test subprocesses must run in
`bwrap --unshare-net`; only the host-side local proxy and stub share that isolated
loopback namespace.

The Aider Polyglot snapshot used for the offline checks was commit
`7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`. Of 225 imported exercises, the
three-gate validated manifest contained 34 Python, 36 Go, and 22 C++ tasks (92
total). Three Go starters already passed their hidden tests; one C++ reference
requires Boost, and three C++ tasks lack references. Rust has 30 reference files
but no local Cargo executable. JavaScript and Java have no reference files in
this Aider snapshot.

## Import and validate Aider Polyglot

With the official Aider Polyglot checkout already in tmpfs:

```sh
nice -n 10 python3 -B bench/agentic-eval/external/prepare_polyglot_tasks.py \
  --source /dev/shm/bench/polyglot \
  --out /dev/shm/bench/tasks-polyglot \
  --languages python,javascript,go,java,cpp,rust
```

Then run the unchanged validator and save the list of usable tasks to tmpfs:

```sh
SHURA_POLYGLOT_NETNS=1 PYTHONDONTWRITEBYTECODE=1 \
POLYGLOT_TMPDIR=/dev/shm/bench/tmp \
POLYGLOT_NPM_MODULES=/dev/shm/bench/npm-install/node_modules \
nice -n 10 bwrap --die-with-parent --unshare-net \
  --ro-bind /usr /usr --ro-bind /etc /etc --proc /proc --dev /dev \
  --tmpfs /tmp --bind /dev/shm/bench /dev/shm/bench \
  --ro-bind /home/alisher/Projects/shura /home/alisher/Projects/shura \
  --chdir /home/alisher/Projects/shura \
  --setenv SHURA_POLYGLOT_NETNS 1 \
  --setenv PATH /dev/shm/bench/aider-venv/bin:/usr/bin:/bin \
  --setenv PYTHONDONTWRITEBYTECODE 1 \
  --setenv PYTEST_DISABLE_PLUGIN_AUTOLOAD 1 \
  --setenv POLYGLOT_TMPDIR /dev/shm/bench/tmp \
  --setenv POLYGLOT_NPM_MODULES /dev/shm/bench/npm-install/node_modules \
  python3 -B bench/agentic-eval/external/validate_polyglot_tasks.py \
  --tasks /dev/shm/bench/tasks-polyglot \
  --manifest /dev/shm/bench/validated-polyglot-ids.txt
```

The command validates that each selected starter fails, its reference passes
three times, tests complete within 30 seconds, and test filenames are absent from
the task prompt. Only IDs in `validated-polyglot-ids.txt` should be used in an
experiment. It skips tasks without an upstream `.meta/example.*` reference and
tasks whose local test runtime is missing.

Reference-source audit for the imported Aider slugs: the official Java track at
`exercism/java@1075bfdabef69af7be24390337d212f179cad9d8` has reference files under
`exercises/practice/<slug>/.meta/src/reference/java/` for all 47 Java exercises.
They are not in the validated manifest because this host has no Gradle or JUnit
cache and the wrapper would need to download packages. The official JavaScript
track at `exercism/javascript@65a60cac2145e8c6549de39e807753e577a02f8d` has no
reference-source files for the 49 matching slugs; its practice folders contain
starters and tests only. The host also has no preinstalled Jest. No source or test
packages were installed.

The ShuraCode host's OpenCode overlay is 1.18.32. In that tag,
`session/llm/request.ts` calls `maxOutputTokens(model, flags.outputTokenMax)`;
`provider/transform.ts` defines `OUTPUT_TOKEN_MAX = 32_000` and clamps with
`Math.min(model.limit.output, outputTokenMax)`. `effect/runtime-flags.ts` reads
`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` as a positive integer. There is no JSON
config setting that raises this engine ceiling. The exact launch environment line
for a future live run is:

```sh
export OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=32768
```

It must be present in the environment of the OpenCode/ShuraCode process. Setting
only `limit.output: 32768` leaves the engine's default ceiling at 32,000.

## Run ShuraCode through the measured harness

First make a scratch config snapshot with `snapshot_live_config.py`. It reads the
current config but writes only to tmpfs; external MCP and personal memory are
excluded. Prepare a validated ID manifest there, then run both complete arms with
the packaged wrapper:

```sh
bash bench/agentic-eval/external/run_tiel_ab.sh /dev/shm/bench --stub
```

The wrapper runs first `tiel-no-plugins`, then
`tiel-all-plugins` (`completion-gate,soft-edit,loop-detector,task-card`) against
the same task manifest and output directory. For a one-task smoke, write just
`aider-python-proverb` to `validated-ids.txt`. `--resume` is on by default, so a
repeated arm skips task IDs already recorded in `results.jsonl`.

For a GPU run, the exact command is:

```sh
bash bench/agentic-eval/external/run_tiel_ab.sh /dev/shm/bench \
  --server-argv-json /dev/shm/bench/llama-argv.json
```

That JSON file is a JSON array containing a test-only
server command with a test port, copied from the approved launch command, and
must include `--spec-type draft-mtp` (the runner refuses to start without it). It
and all outputs stay in tmpfs. The runner sets model output limit 32,768, context
131,072, and `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=32768`; it captures each raw
OpenAI-compatible request body into its own atomically renamed gzip JSON file
before proxy rewriting, so a stopped run cannot corrupt earlier request captures
and the actual `max_tokens` is verifiable. S6 `safety.py` watchdog is active for real
runs; the stub path feeds synthetic safe readings and never opens NVML. Launch
with `taskset -c 0,1 nice -n 19` to keep all child processes at the requested
priority and affinity.

The one-task stub smoke is about 2–3 seconds after OpenCode dependency checks are
seeded from the preinstalled read-only SDK. Using the prior S6 observation of
about six minutes per real task, 92 tasks take roughly 9.2 hours per arm, or 18.4
hours total, before restart/test overhead. A 40-task subset is about four hours
per arm/eight hours total; a 50-task subset is about five hours per arm/ten hours
total. No real model was started as part of this offline task.

Paired-power simulation for the single two-arm comparison (80 Monte Carlo trials,
1,000 sign flips, one planned contrast) gave, under the broad task-difficulty mix,
approximate 5/10-point power of
2.5%/13.8% for 40 tasks and 5.0%/18.8% for 50 tasks. The 80%-power minimum
detectable effect was about 24.6 and 23.9 percentage points respectively. These
small subsets therefore cannot reliably resolve a 5–10-point plugin effect.
For all 92 validated tasks, the same simulation estimated 12.5%/36.2% power at
5/10 points and an 80%-power detectable effect of about 16.2 points. Separately
testing four plugin contrasts and applying Holm correction would lower power
further.

## Reference-only cycles

These do not measure ShuraCode. They are only transport and leaderboard sanity
checks. The tested Aider command is implemented in `smoke_aider.py`; a one-task
smoke took about one second of model/test work after Python startup. For the full
225-task official cycle, with the benchmark-root symlink and model-settings file
created by `smoke_aider.py`, run:

```sh
OPENAI_API_KEY=stub-or-local \
OPENAI_API_BASE=http://127.0.0.1:8090/v1 \
AIDER_DOCKER=1 \
AIDER_BENCHMARK_DIR=/dev/shm/bench/aider/benchmark-root \
AIDER_MODEL_METADATA_FILE=/dev/shm/bench/aider/model-metadata.json \
nice -n 10 /dev/shm/bench/aider-venv/bin/python \
  /dev/shm/bench/aider-src/benchmark/benchmark.py shura-reference \
  --model openai/shura-tiel --edit-format whole \
  --languages python,javascript,go,java,cpp,rust \
  --num-tests 225 --tries 1 --threads 1 \
  --read-model-settings /dev/shm/bench/aider/model-settings.yml \
  --exercises-dir polyglot-benchmark --new
```

Use the Occamy settings file and model ID for Occamy. Actual wall time depends on
model output and is expected to be several hours; the stub smoke is not a timing
proxy. Do not compare this model-only score directly to the ShuraCode harness score.

BFCL's tested command is implemented in `smoke_bfcl.py`: it uses the pinned
`bfcl-eval==2025.12.17` package and `--skip-server-setup --run-ids` to restrict the
smoke to one simple-Python and one multi-turn case. A full simple-Python plus
multi-turn-base reference run covers 600 cases. With a local llama-server and the
matching environment variables from `smoke_bfcl.py`, run:

```sh
BFCL_PROJECT_ROOT=/dev/shm/bench/bfcl/run \
OPENAI_BASE_URL=http://127.0.0.1:8090/v1 OPENAI_API_KEY=stub-or-local \
SHURA_TOP_P=0.95 SHURA_TOP_K=20 SHURA_MAX_TOKENS=32768 \
nice -n 10 /dev/shm/bench/bfcl-venv/bin/bfcl generate \
  --model shura-tiel --test-category simple_python,multi_turn_base \
  --skip-server-setup --num-threads 1 --temperature 0.6

BFCL_PROJECT_ROOT=/dev/shm/bench/bfcl/run \
nice -n 10 /dev/shm/bench/bfcl-venv/bin/bfcl evaluate \
  --model shura-tiel --test-category simple_python,multi_turn_base --partial-eval
```

Use Occamy temperature `1.0` and `SHURA_PRESENCE_PENALTY=1.5` for Occamy. The
reference run covers 600 cases and will take several hours on a local model. BFCL
results are also reference-only.

Run the behavior summary after an experiment:

```sh
python3 -B bench/agentic-eval/external/behavior_metrics.py \
  --results /dev/shm/bench/run/results.jsonl \
  --agent-logs /dev/shm/bench/run/agent-logs \
  --out /dev/shm/bench/run/behavior-summary.json
```

`power_external.py` uses the same paired sign-flip model as `power.py`; its output
assumes independent tasks and applies Holm correction for four planned contrasts.
