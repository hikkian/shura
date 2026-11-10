#!/bin/bash
# OCR path for the picture task: ornith5 (ideal5) + prompt-v5 (one extra rule: read exact text/digits from images with tesseract) on h4 x3 at xhigh.
R=/home/hikkian/Desktop/AI/bench/ruspell; H=/home/hikkian/Desktop/AI/shura-report; P=/home/hikkian/.config/shuracode/prompt.txt
G=/home/hikkian/Desktop/AI/bench/template/gen10
while [ ! -f $R/STAGE24_DONE ]; do sleep 30; done
A=arm_ornith5_xhigh_ocr; cp $G/prompt-v5.txt $P; systemctl --user restart ai-gateway; sleep 20; cd $H
: > results/$A.jsonl; echo "$A start $(date +%T)"
for r in 1 2 3; do ./bench.sh $A h4_invoice_mockup $r >> results/$A-batch.log 2>&1 < /dev/null; echo "$A h4 rep$r $(date +%T)"; done
python3 rescore.py $A > results/rescore-$A.log 2>&1
cp $P.bak-20261108-honesty $P; systemctl --user restart ai-gateway
touch $R/STAGE25_DONE
