# Adding your hardware, a model or a backend

Shura is meant to run on hardware its author does not own. That only works if people like you add what they
know. Everything is data first, code second, and every step has a test, so a small contribution is enough.
(The logic behind the decisions is in [docs/HARDWARE.md](docs/HARDWARE.md).)

## 1. The fastest way to help: a hardware report

```bash
shura report        # writes shura-hardware-report.md: no names, paths or IDs, nothing is sent
```

Open a *Hardware report* issue and paste it, plus what speed you really saw. Reports are how a machine class goes
from "unverified" to "verified" in the status table.

## 2. Add a machine profile (a test fixture)

1. Take the JSON block from your report (or copy a file from `tests/fixtures/hardware/`).
2. Save it as `tests/fixtures/hardware/<gpu>_<vram>g_<ram>g.json`. Units are bytes; `bandwidth_gbs` should be measured.
3. Run `python3 -m unittest tests.test_universal`. If the planner's choice for your machine is wrong, open the PR
   anyway and say what you measured: a wrong plan with a real measurement is a bug report we can act on.

## 3. Add a model to the catalog

1. Copy an entry in `catalog/models.json`; fill in architecture, parameters, `kv_bytes_per_token_f16`, quants with
   exact `file_bytes`, and `nonexpert_bytes` (for MoE: the part that is not routed experts; the installer's GGUF reader
   can print it).
2. `python3 -m unittest tests.test_universal` validates the schema and lists every problem at once.
3. Add a `tested` entry (hardware, backend, result, date). An entry without one is shown as unverified.
4. MoE models also need `expert_active_fraction`; if you do not know it, say so in the PR and use 0.3 as a start.

## 4. Add or fix a backend

A backend needs: the asset pattern of its prebuilt llama.cpp build (`installer/universal/backends.py`), a rule for when it
is a candidate for a GPU (`planner.BACKEND_ORDER`), and a self-test that passes: `python3 installer/universal/cli.py
selftest --backend <name>`. Backends we can test without hardware run in CI (CPU on Linux, macOS and Windows; Vulkan
on Mesa's software driver). Those that cannot (CUDA, ROCm, SYCL, Metal) rely on your report.

## 5. Before you open a pull request

```bash
python3 -m unittest discover -s tests                       # everything, offline
ruff check gateway bench installer tests --select E,F,W,B --ignore E501
shellcheck setup.sh scripts/*.sh scripts/shura tests/*.sh
```

Tests must not assume your machine: no GPU, no NVIDIA libraries and a disk `/tmp` on CI. If you use an AI coding agent
(ShuraCode works too), point it at this file and at `docs/HARDWARE.md`; the checklist above is written so that an agent
can follow it.
