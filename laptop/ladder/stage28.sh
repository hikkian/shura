#!/bin/bash
# Vision shootout on the picture task (runs after stage22). Candidates are all projector files we have that fit the 35B-A3B language model:
# A current (Ornith Q8_0), B Ornith BF16, C Qwen3.6 Q8_0, D Sage-A1.5 projector, E the full Qwen3.6 model (own language part). Same ideal5 template, thinking off, 4 samples each.
R=/home/hikkian/Desktop/AI/bench/ruspell; OUT=/home/hikkian/Desktop/AI/mgmt-report/vision
M=/home/hikkian/.local/share/shura/models
while [ ! -f $R/STAGE27_DONE ]; do sleep 30; done
echo "stage28 (shootout rerun 2) setup $(date +%T)"
cp /home/hikkian/shura/config/model-launch.json /home/hikkian/shura/config/model-launch.json.bak-stage24
cp /home/hikkian/Desktop/AI/bench/template/gen8/qwen36-ideal5.jinja /home/hikkian/shura/config/templates/qwen36-ideal5.jinja
python3 - <<'PY'
import json
M='/home/hikkian/shura/config/model-launch.json'; d=json.load(open(M)); base='/home/hikkian/.local/share/shura/models/'
def clone(name, mm):
    m=json.loads(json.dumps(d['models']['ornith5'])); m['mmprojPath']=base+mm; d['models'][name]=m
clone('visA','mmproj-Q8_0.gguf'); clone('visB','mmproj-Ornith-1.5-35B-BF16.gguf'); clone('visC','cand/qwen36-mmproj-Q8_0.gguf'); clone('visD','cand/Sage-A1.5-mmproj.gguf')
m=json.loads(json.dumps(d['models']['qwen36i']).replace('qwen36-ideal.jinja','qwen36-ideal5.jinja')); m['langGuard']=True; d['models']['qwen36v5']=m
json.dump(d,open(M,'w'),indent=2,ensure_ascii=False); print('vision candidates configured')
PY
systemctl --user restart ai-gateway; sleep 20
: > $OUT/shootout.jsonl
for mod in visA visB visC visD qwen36v5; do echo "$mod $(date +%T)"; python3 $OUT/shootout.py $mod 4 >> $OUT/shootout.jsonl 2>> $OUT/shootout.err; tail -1 $OUT/shootout.jsonl | cut -c1-160; done
touch $R/STAGE28_DONE
