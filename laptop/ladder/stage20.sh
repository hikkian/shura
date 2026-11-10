#!/bin/bash
# Codex (gpt-6-luna, xhigh as set by the user in ~/.codex/config.toml) on the same new tasks as the local arms:
# hd1..hd6 x2 reps (same set as the effort ladder) and h1..h5 x3 reps (same set as the template comparison). One run at a time.
H=/home/hikkian/Desktop/AI/shura-report; cd $H
echo "stage20 start $(date +%T)"
for r in 1 2; do for t in hd1_shift_hours hd2_js_cache hd3_calc_parser hd4_recurrence hd5_patch_apply hd6_js_allocate; do
  ./bench.sh codex $t $r >> results/codex-new-batch.log 2>&1 < /dev/null; echo "codex $t $r $(date +%T)"; done; done
for r in 1 2 3; do for t in h1_shop_discount h2_js_pool h3_readme_envconf h4_invoice_mockup h5_ci_cause; do
  ./bench.sh codex $t $r >> results/codex-new-batch.log 2>&1 < /dev/null; echo "codex $t $r $(date +%T)"; done; done
touch /home/hikkian/Desktop/AI/bench/ruspell/STAGE20_DONE
