#!/bin/bash
# ornith7 = ideal7 (constrained magnification, first-version-early) + per-step caps + safe pkill + Q8_0 projector (BF16 did not help).
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10
TASKS="h1_shop_discount h3_readme_envconf h4_invoice_mockup h5_ci_cause"
while [ ! -f $R/STAGE17_DONE ]; do sleep 30; done
echo "stage19 setup $(date +%T)"
cp $G/ornith-ideal7.jinja /home/hikkian/shura/config/templates/ornith-ideal7.jinja
cp /home/hikkian/shura/config/model-launch.json /home/hikkian/shura/config/model-launch.json.bak-stage19
cp /home/hikkian/.config/shuracode/shuracode.json /home/hikkian/.config/shuracode/shuracode.json.bak-stage19
python3 - <<'PY'
import json, copy
M='/home/hikkian/shura/config/model-launch.json'; d=json.load(open(M))
m=json.loads(json.dumps(d['models']['ornith6']).replace('ornith-ideal6.jinja','ornith-ideal7.jinja'))
m['mmprojPath']=d['models']['ornith5']['mmprojPath']
d['models']['ornith7']=m
json.dump(d,open(M,'w'),indent=2,ensure_ascii=False)
S='/home/hikkian/.config/shuracode/shuracode.json'; s=json.load(open(S))
e=copy.deepcopy(s['provider']['tiel-local']['models']['ornith6']); e['name']='Ornith-1.5 35B-A3B effort v7 (test)'
s['provider']['tiel-local']['models']['ornith7']=e
json.dump(s,open(S,'w'),indent=2,ensure_ascii=False); print('ornith7 configured')
PY
A=arm_ornith7_xhigh_ho; cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
echo "$A start $(date +%T)"; : > results/$A.jsonl
for r in 1 2 3; do for t in $TASKS; do ./bench.sh $A $t $r >> results/$A-batch.log 2>&1 < /dev/null; done; done
python3 rescore.py $A > results/rescore-$A.log 2>&1; echo "$A end $(date +%T): $(tail -1 results/rescore-$A.log)"
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('/home/hikkian/Desktop/AI/shura-report/results/arm_ornith7_xhigh_ho.jsonl')]
ok=sum(1 for r in rows if r['pass']); to=sum(1 for r in rows if r['secs']>=1500)
m='ornith7' if (ok>=5 and to<=2) else 'ornith5'
open('/home/hikkian/Desktop/AI/bench/ruspell/LADDER_MODEL','w').write(m)
print(f'decision: ornith7 passed {ok}/{len(rows)}, timeouts {to} -> ladder on {m}')
PY
touch $R/STAGE19_DONE
