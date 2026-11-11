#!/bin/bash
# Single-layer scan 28-38 (expert gate/up/down of ONE layer to pure Q4_K), then a combined variant of the best layers + 39. Runs after stage31.
W=/home/hikkian/Desktop/AI/weights-fix; R=/home/hikkian/Desktop/AI/bench/ruspell; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python
while [ ! -f $R/STAGE31_DONE ]; do sleep 30; done
systemctl --user stop ai-gateway; sleep 6
echo "single scan start $(date +%T)"
ALL=ffn_gate_exps,ffn_up_exps,ffn_down_exps
for L in 38 37 36 35 34 33 32 31 30 29 28; do
  O=$W/mix-s$L.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K-pure.gguf --out $O --layers $L --kinds $ALL > $W/splice-s$L.log 2>&1 || { echo "splice s$L failed: $(tail -1 $W/splice-s$L.log)"; continue; }
  $W/eval_variant.sh $O s$L; rm -f $O
done
# pick: best two layers by margin, requiring ppl not worse than base (5.734) and code margin >= 5.1
read A B < <(python3 - <<'PY'
import json,re
rows=[]
for l in open('/home/hikkian/Desktop/AI/weights-fix/results.jsonl'):
    d=json.loads(l)
    if re.fullmatch(r's\d+',d['label']):
        p=float(re.search(r'PPL = ([0-9.]+)',d['ppl']).group(1)); m=d['margin']
        if p<=5.734 and m['code_mean_margin']>=5.1: rows.append((m['mean_margin'],int(d['label'][1:])))
rows.sort(reverse=True); L=[x[1] for x in rows[:2]]+[0,0]; print(L[0],L[1])
PY
)
echo "best layers: $A $B"
if [ "$A" != "0" ]; then
  LAY=39,$A; [ "$B" != "0" ] && LAY=39,$A,$B
  O=$W/mix-multi.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K-pure.gguf --out $O --layers $LAY --kinds $ALL > $W/splice-multi.log 2>&1 && { grep Итого $W/splice-multi.log | cut -c1-120; $W/eval_variant.sh $O multi-$LAY; }
  [ -f $O ] && mv $O /home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-multi.gguf
fi
systemctl --user start ai-gateway
touch $W/SINGLE_DONE
