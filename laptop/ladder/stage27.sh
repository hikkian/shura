#!/bin/bash
# Automatic OCR in the ShuraCode plugin (chat.message hook: attached images are read by tesseract after a 2x upscale, the text is added to the
# message). Test: h4 x3 at xhigh on ornith5 + plain prompt-v4 (no OCR rule in the prompt). The plugin is installed for the test and the backup
# restored afterwards; whether to keep it is decided from the numbers.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10; PL=/home/hikkian/.local/share/shura/shuracode/plugin/server.ts
while [ ! -f $R/STAGE26_DONE ]; do sleep 30; done
cp $PL /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup; cp /home/hikkian/Desktop/AI/shura-ocr/server.ts $PL
A=arm_ornith5_xhigh_ocrplug; cp $G/prompt-v4.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
: > results/$A.jsonl; echo "$A start $(date +%T)"
for r in 1 2 3; do ./bench.sh $A h4_invoice_mockup $r >> results/$A-batch.log 2>&1 < /dev/null; echo "$A h4 rep$r $(date +%T)"; done
python3 rescore.py $A > results/rescore-$A.log 2>&1
cp /home/hikkian/Desktop/AI/shura-ocr/server.ts.production-backup $PL
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE27_DONE
