#!/bin/bash
# Second repetition for the top levels with ONE equal limit (3000 s) for both; hd5 is left out of the repetition (every level times out on it,
# it costs ~50 min per run and says nothing about the ladder; it stays in the report as its own row). Order: task outer, level inner,
# so a cut-off leaves xhigh and max with equal data. Rows are appended to the same arm files; runs already done are skipped.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10
TASKS="hd1_shift_hours hd2_js_cache hd3_calc_parser hd4_recurrence hd6_js_allocate"
cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
export BENCH_TIMEOUT=3000
echo "stage22 (v2) start $(date +%T)"
for t in $TASKS; do for L in xhigh max; do A=arm_ornith5_${L}_lad
  if python3 -c "import json,sys; sys.exit(0 if any(json.loads(l)['task']=='$t' and json.loads(l)['rep']==2 for l in open('results/$A.jsonl')) else 1)"; then echo "skip $A $t (done)"; continue; fi
  ./bench.sh $A $t 2 >> results/$A-batch.log 2>&1 < /dev/null; echo "$A $t rep2 $(date +%T)"; done; done
for A in arm_ornith5_xhigh_lad arm_ornith5_max_lad; do python3 rescore.py $A > results/rescore-$A.log 2>&1; done
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE22_DONE
