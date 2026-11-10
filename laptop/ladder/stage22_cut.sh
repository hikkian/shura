#!/bin/bash
# Time budget: the picture tests must finish by ~19:00, so the second repetition of xhigh/max is cut after hd3 (n=9 per level instead of 11).
# Decided in advance (before seeing those results): cut point = after "max hd3 rep2". Then finalize stage22 the way its own tail would.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
until grep -q "^arm_ornith5_max_lad hd3_calc_parser rep2" $R/stage22b.log; do sleep 2; done
PG=$(pgrep -f '^/bin/bash ./stage22.sh$' | head -1)
[ -n "$PG" ] && kill -TERM -- -$PG 2>/dev/null; sleep 3
echo "stage22 cut after hd3 at $(date +%T)" >> $R/stage22b.log
cd $H; for A in arm_ornith5_xhigh_lad arm_ornith5_max_lad; do python3 rescore.py $A > results/rescore-$A.log 2>&1; done
touch $R/STAGE22_DONE
