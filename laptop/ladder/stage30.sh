#!/bin/bash
# Ornith (ornith5 = ideal5, prompt-v4) + plugin OCR on h4 at medium and high, 3 reps each: does a lower effort keep the quality and cut the time (Qwen3.6 needs ~2 min)?
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10; PL=/home/hikkian/.local/share/shura/shuracode/plugin/server.ts
while [ ! -f $R/STAGE29_DONE ]; do sleep 20; done
cp $PL /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup3; cp /home/hikkian/Desktop/AI/shura-ocr/server.ts $PL
cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
for L in medium high; do A=arm_ornith5_${L}_ocrplug; : > results/$A.jsonl; echo "$A start $(date +%T)"
  for r in 1 2 3; do ./bench.sh $A h4_invoice_mockup $r >> results/$A-batch.log 2>&1 < /dev/null; echo "$A h4 rep$r $(date +%T)"; done
  python3 rescore.py $A > results/rescore-$A.log 2>&1; done
cp /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup3 $PL
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE30_DONE
