#!/bin/bash
# After the ladder's first repetition (fast..max x hd1-6) finishes: stop stage18 (before its 2nd repetition) and spend the GPU on the question
# "is high < medium real, and is the long-step budget the cause?":
#   (a) rep 2 of medium and high on hd1-6 (noise check, same arms/same files, rows appended)
#   (b) arm ornith8 = ornith5 + effortStepCaps {high: 8192} (steps after the first get the medium-like cap), high x hd1-6, 1 rep
# Frozen in advance: no other criteria; results are reported as measured.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
TASKS="hd1_shift_hours hd2_js_cache hd3_calc_parser hd4_recurrence hd5_patch_apply hd6_js_allocate"
until grep -q "^arm_ornith5_max_lad rep1 end" $R/stage18.log; do sleep 2; done
PG=$(pgrep -f '^/bin/bash ./stage18.sh$' | head -1)
[ -n "$PG" ] && kill -TERM -- -$PG 2>/dev/null; sleep 3; pkill -f "shuracode run --auto -m tiel-local/ornith5" 2>/dev/null
echo "stage18 stopped after rep1 $(date +%T)"
python3 - <<'PY'
import json, copy
M='/home/hikkian/shura/config/model-launch.json'; d=json.load(open(M))
m=json.loads(json.dumps(d['models']['ornith5'])); m['effortStepCaps']={'high':8192}
d['models']['ornith8']=m; json.dump(d,open(M,'w'),indent=2,ensure_ascii=False)
S='/home/hikkian/.config/shuracode/shuracode.json'; s=json.load(open(S))
e=copy.deepcopy(s['provider']['tiel-local']['models']['ornith5']); e['name']='Ornith-1.5 35B-A3B v5 + step cap (test)'
s['provider']['tiel-local']['models']['ornith8']=e; json.dump(s,open(S,'w'),indent=2,ensure_ascii=False); print('ornith8 configured')
PY
systemctl --user restart ai-gateway; sleep 20; cd $H
for L in medium high; do A=arm_ornith5_${L}_lad; echo "$A rep2 start $(date +%T)"
  for t in $TASKS; do ./bench.sh $A $t 2 >> results/$A-batch.log 2>&1 < /dev/null; done; echo "$A rep2 end $(date +%T)"; done
A=arm_ornith8_high_lad; : > results/$A.jsonl; echo "$A rep1 start $(date +%T)"
for t in $TASKS; do ./bench.sh $A $t 1 >> results/$A-batch.log 2>&1 < /dev/null; done; echo "$A rep1 end $(date +%T)"
for A in arm_ornith5_fast_lad arm_ornith5_low_lad arm_ornith5_medium_lad arm_ornith5_high_lad arm_ornith5_xhigh_lad arm_ornith5_max_lad arm_ornith8_high_lad; do python3 rescore.py $A > results/rescore-$A.log 2>&1; done
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE21_DONE
