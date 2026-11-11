#!/bin/bash
# After the typo scan: decide prompt v4/v6 by the frozen rule (DECISION-RULE-v6.txt), put the chosen prompt into production, then run the 19 old tasks on the
# production configuration at medium, repetition outer / task inner, until 08:45 (finalization window 08:45-10:00).
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt; G=/home/hikkian/Desktop/AI/bench/template; W=/home/hikkian/Desktop/AI/weights-fix
while [ ! -f $W/SINGLE_DONE ]; do sleep 30; done
CHOICE=$(python3 - <<'PY'
import json,re
def frac(r):
    if r['hidden']: return 1.0
    m=re.search(r'Ran (\d+) tests?',r['hidden_err']); f=re.search(r'FAILED \(([^)]*)\)',r['hidden_err'])
    if not m: return 0.0
    n=int(m.group(1)); bad=sum(int(x) for x in re.findall(r'=(\d+)',f.group(1))) if f else 0
    return (n-bad)/n
def load(a): 
    try: return [json.loads(l) for l in open(f'/home/hikkian/Desktop/AI/shura-report/results/{a}.jsonl')]
    except Exception: return []
m=lambda rs: sum(frac(r) for r in rs)/len(rs) if rs else None
h4,h6=load('arm_ornithi_high_p4'),load('arm_ornithi_high_p6'); d4,d6=load('arm_ornithi_medium_p4'),load('arm_ornithi_medium_p6')
ok=False
if h4 and h6 and d4 and d6:
    h_gain=m(h6)-m(h4); passes=sum(r['pass'] for r in h6)>sum(r['pass'] for r in h4); d_loss=m(d4)-m(d6)
    ok=(h_gain>=0.10 or passes) and d_loss<=0.05
    print('v6' if ok else 'v4', f'h1 p4 {m(h4):.2f} p6 {m(h6):.2f} | hd p4 {m(d4):.2f} p6 {m(d6):.2f}', file=__import__('sys').stderr)
print('v6' if ok else 'v4')
PY
)
echo "prompt decision: $CHOICE ($(date +%T))"
[ "$CHOICE" = "v6" ] && { cp -p $P $P.bak-before-v6; cp $G/gen11/prompt-v6.txt $P; }
cd $H; A=arm_ornithi_medium_f19; : > results/$A.jsonl
TASKS=$(ls /home/hikkian/Desktop/AI/shuracode-prompt/eval/tasks | grep -E "^t")
for rep in 1 2 3; do for t in $TASKS; do
  [ $(date +%H%M | sed 's/^0*//') -ge 845 ] && [ $(date +%H) -lt 12 ] && { echo "cutoff reached $(date +%T)"; touch $R/STAGE33_DONE; exit 0; }
  ./bench.sh $A $t $rep >> results/$A-batch.log 2>&1 < /dev/null; echo "$A $t $rep $(date +%T)"
done; done
touch $R/STAGE33_DONE
