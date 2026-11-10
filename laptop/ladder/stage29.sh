#!/bin/bash
# Qwen3.6 (ideal5, prompt-v4) with the plugin OCR on h4 x3 at xhigh: the best "picture" combination candidate. Plugin installed for the test only.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10; PL=/home/hikkian/.local/share/shura/shuracode/plugin/server.ts
cp $PL /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup2; cp /home/hikkian/Desktop/AI/shura-ocr/server.ts $PL
A=arm_qwen36v5_xhigh_ocrplug; cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
: > results/$A.jsonl; echo "$A start $(date +%T)"
for r in 1 2 3; do ./bench.sh $A h4_invoice_mockup $r >> results/$A-batch.log 2>&1 < /dev/null; echo "$A h4 rep$r $(date +%T)"; done
python3 rescore.py $A > results/rescore-$A.log 2>&1
cp /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup2 $PL
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE29_DONE
