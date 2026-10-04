# External quality benchmarks

This directory keeps the public-reference pipeline separate from the v1/v2 task
sets. Use a `/dev/shm` workspace for source clones, imported exercises, generated
tasks, sessions, caches, and results. The agent and test subprocesses must run in
`bwrap --unshare-net`; only the host-side local proxy and stub share that isolated
loopback namespace.

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
experiment. It skips tasks without an upstream `.meta/example.*` reference.

## Run ShuraCode through the measured harness

First make a scratch config snapshot with `snapshot_live_config.py`. It reads the
current config but writes only to tmpfs; external MCP and personal memory are
excluded. Then run a single-task smoke against the local stub:

```sh
SHURA_POLYGLOT_NETNS=1 nice -n 10 bwrap --die-with-parent --unshare-net \
  ... bind the ShuraCode engine read-only, the plugin SDK read-only, and `/dev/shm/bench` ... \
  python3 -B bench/agentic-eval/external/polyglot_harness.py \
  --tasks /dev/shm/bench/tasks-polyglot \
  --out /dev/shm/bench/smoke \
  --config-snapshot /dev/shm/bench/config.snapshot.json \
  --scratch /dev/shm/bench/scratch --model tiel \
  --languages python --task-ids aider-python-proverb --timeout-s 30 --stub
```

For real-model runs, replace `--stub` with a JSON array in a tmpfs file passed
through `--server-argv-json`; use a test-only server command copied from the
approved launch configuration, with only its port and slot path redirected to
tmpfs. Run the measured proxy and agent in the same no-network namespace. Tiel
uses `0.6 / 0.95 / 20`, Occamy `1.0 / 0.95 / 20 / presence_penalty 1.5`, both
with a 32,768 output limit. The runner saves one JSONL result per task and keeps
each raw agent log in the selected output directory.

The one-task stub smoke is about 2–3 seconds after OpenCode dependency checks are
seeded from the preinstalled read-only SDK. A real 92-task pass will take on the
order of hours and must be scheduled as a separate, approved GPU run; no real
model was started as part of this offline task.

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
