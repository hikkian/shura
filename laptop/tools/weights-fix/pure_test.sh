#!/bin/bash
# Layer 39 with a pure Q4_K donor (down tensor Q4_K instead of Q6_K): memory comfort + quality.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python
OUT=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf
until grep -q "total time" $W/donor-pure.log; do sleep 15; done
$PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K-pure.gguf --out $OUT --layers 39 --kinds ffn_gate_exps,ffn_up_exps,ffn_down_exps > $W/splice-l39p.log 2>&1 || { echo "splice failed"; exit 1; }
grep "Итого" $W/splice-l39p.log | cut -c1-120
python3 - <<'PY'
import json
p='/home/hikkian/shura/config/model-launch.json'; d=json.load(open(p))
m=json.loads(json.dumps(d['models']['ornithi'])); m['modelPath']='/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP-l39p.gguf'
d['models']['ornithl39p']=m; json.dump(d,open(p,'w'),indent=2,ensure_ascii=False)
PY
systemctl --user start ai-gateway; sleep 8
python3 $W/measure_comfort.py ornithi ornithl39p ornithi
systemctl --user stop ai-gateway; sleep 6
$W/eval_variant.sh $OUT l39p
systemctl --user start ai-gateway
touch $W/PURE_DONE
