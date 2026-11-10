#!/bin/bash
# Effort ladder: one model, six levels (fast low medium high xhigh max), the 6 hard reasoning tasks hd1..hd6, 2 repetitions.
# Order: repetition outer, level inner -> if the night is cut short the levels still have equal data; drift (heat, time) is spread across levels.
# Frozen BEFORE the run: templates, budgets, step caps, prompt. Nothing is tuned on these results.
# usage: stage18.sh <model> <prompt-file>      e.g. stage18.sh ornith6 /home/hikkian/Desktop/AI/bench/template/gen9/prompt-v4.txt
PROMPT=${2:-/home/hikkian/Desktop/AI/bench/template/gen10/prompt-v4.txt}
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
TASKS="hd1_shift_hours hd2_js_cache hd3_calc_parser hd4_recurrence hd5_patch_apply hd6_js_allocate"
LEVELS="fast low medium high xhigh max"
while [ ! -f $R/STAGE19_DONE ]; do sleep 30; done
MODEL=$(cat $R/LADDER_MODEL)   # chosen by the pre-registered rule in DECISION-RULE.txt
echo "stage18 start $(date +%T) model=$MODEL"
cp $PROMPT $P; systemctl --user restart ai-gateway; sleep 20; cd $H
for L in $LEVELS; do : > results/arm_${MODEL}_${L}_lad.jsonl; done
for r in 1 2; do for L in $LEVELS; do A=arm_${MODEL}_${L}_lad
  echo "$A rep$r start $(date +%T)"
  for t in $TASKS; do ./bench.sh $A $t $r >> results/$A-batch.log 2>&1 < /dev/null; done
  echo "$A rep$r end $(date +%T)"
done; done
for L in $LEVELS; do A=arm_${MODEL}_${L}_lad; python3 rescore.py $A > results/rescore-$A.log 2>&1; echo "$A: $(tail -1 results/rescore-$A.log)"; done
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE18_DONE
