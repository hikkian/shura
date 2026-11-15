# Compatibility: what has been run, and on what

This table is the pilot programme's record: one row per machine that someone actually ran Shura on. Rows are added from
`shura report` files with [`scripts/compat_table.py`](../scripts/compat_table.py). Nothing is judged by hand except the
name of who ran it and the date.

**How to join.** Run `./scripts/shura install` (or `./scripts/shura check` if you only want the plan), then
`./scripts/shura report`, and open a "Pilot report" issue (label `pilot`) with the file attached. A maintainer adds the row.

**Status**

- ✓ The server ran with the planned settings, and its measured speed was within 25 % of the installer's estimate.
- ⚠ It runs, but the measured speed is outside that band, or the install did not measure the speed (a plan only).
- ✗ It did not work: no plan fits the machine, or the install could not verify the server.

**Speed** is the measured short-prompt speed when the installer measured it ("замер"); otherwise the estimate ("оценка").
The estimate for MoE models on Vulkan is not calibrated yet ([SPEED-ESTIMATES.md](SPEED-ESTIMATES.md)), so a ⚠ there is
expected until a measurement is in.

<!-- compat:start -->
| Карта | Драйвер | ОС | Бэкенд | Модель | Скорость, tok/s | Статус | Кто проверил | Дата |
|---|---|---|---|---|---|---|---|---|
| RX 7800 XT (16 GB) | ROCm 7.1.1 (HIP, our fork 95d26d1) | Fedora 44, Linux 7.2 | hip (наш форк) | Tiel-Coder 35B-A3B APEX-I-MiniPlus V2.1 | 74 декод (профиль A, окно 26k); 44 при 60k и 38 при 150k (профиль B) | ⚠ | hikkian (own stand) | 2026-10-20 |
<!-- compat:end -->

The first row is our own machine. Its numbers come from the lab reports REPORT-PREFILL3 and REPORT-LAYOUT1c (kept outside
this repository), measured by hand with our own layout, not by the installer. It is ⚠ until the installer runs here.
