#!/bin/bash
# Small-tensor variants (token embedding and output head raised to Q8_0 from the Q8_0 file). Runs after scan_groups.sh finished.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python; Q8=/home/hikkian/Desktop/AI/bench/rufix/src/Ornith-1.5-35B-Q8_0.gguf
while [ ! -f $W/SCAN1_DONE ]; do sleep 20; done
systemctl --user stop ai-gateway; sleep 6
echo "emb scan start $(date +%T)"
for v in "emb8:token_embd.weight" "head8emb8:token_embd.weight,output.weight"; do
  L=${v%%:*}; N=${v#*:}; O=$W/mix-$L.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $Q8 --out $O --names $N > $W/splice-$L.log 2>&1 || { echo "splice $L failed"; continue; }
  $W/eval_variant.sh $O $L; rm -f $O
done
echo "emb scan end $(date +%T)"
systemctl --user start ai-gateway
touch $W/SCAN2_DONE
