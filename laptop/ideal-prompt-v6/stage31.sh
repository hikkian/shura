#!/bin/bash
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt; G=/home/hikkian/Desktop/AI/bench/template
cd $H; for A in arm_ornithi_high_p4 arm_ornithi_high_p6 arm_ornithi_medium_p4 arm_ornithi_medium_p6; do : > results/$A.jsonl; done
run() { A=$1; T=$2; REP=$3; PF=$4; cp $PF $P; echo "$A $T $REP start $(date +%T)"; ./bench.sh $A $T $REP >> results/$A-batch.log 2>&1 < /dev/null; echo "$A $T $REP end $(date +%T)"; }
for rep in 1 2; do
  run arm_ornithi_high_p4 h1_shop_discount $rep $G/gen10/prompt-v4.txt
  run arm_ornithi_high_p6 h1_shop_discount $rep $G/gen11/prompt-v6.txt
done
export BENCH_TIMEOUT=900
for t in hd1_shift_hours hd6_js_allocate; do
  run arm_ornithi_medium_p4 $t 1 $G/gen10/prompt-v4.txt
  run arm_ornithi_medium_p6 $t 1 $G/gen11/prompt-v6.txt
done
cp $G/gen10/prompt-v4.txt $P; touch $R/STAGE31_DONE
