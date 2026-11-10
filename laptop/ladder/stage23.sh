#!/bin/bash
# Picture question: same task h4 (invoice mockup, small footer digits) on the full Qwen3.6 model (ideal5 template, prompt-v4) x3 at xhigh.
# The vision projector files of Ornith and Qwen3.6 are byte-identical (checked), so this tests the language model's use of the image, not the encoder.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10
while [ ! -f $R/STAGE28_DONE ]; do sleep 30; done
echo "stage23 setup $(date +%T)"
cp /home/hikkian/Desktop/AI/bench/template/gen8/qwen36-ideal5.jinja /home/hikkian/shura/config/templates/qwen36-ideal5.jinja
cp /home/hikkian/shura/config/model-launch.json /home/hikkian/shura/config/model-launch.json.bak-stage23
cp /home/hikkian/.config/shuracode/shuracode.json /home/hikkian/.config/shuracode/shuracode.json.bak-stage23
python3 - <<'PY'
import json, copy
M='/home/hikkian/shura/config/model-launch.json'; d=json.load(open(M))
m=json.loads(json.dumps(d['models']['qwen36i']).replace('qwen36-ideal.jinja','qwen36-ideal5.jinja')); m['langGuard']=True
d['models']['qwen36v5']=m; json.dump(d,open(M,'w'),indent=2,ensure_ascii=False)
S='/home/hikkian/.config/shuracode/shuracode.json'; s=json.load(open(S))
e=copy.deepcopy(s['provider']['tiel-local']['models']['qwen36i']); e['name']='Qwen3.6 35B-A3B effort v5 (test)'
s['provider']['tiel-local']['models']['qwen36v5']=e; json.dump(s,open(S,'w'),indent=2,ensure_ascii=False); print('qwen36v5 configured')
PY
A=arm_qwen36v5_xhigh_ho; cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
: > results/$A.jsonl; echo "$A start $(date +%T)"
for r in 1 2 3; do ./bench.sh $A h4_invoice_mockup $r >> results/$A-batch.log 2>&1 < /dev/null; echo "$A h4 rep$r $(date +%T)"; done
python3 rescore.py $A > results/rescore-$A.log 2>&1
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE23_DONE
