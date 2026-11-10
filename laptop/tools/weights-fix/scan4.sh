#!/bin/bash
# Last layers only: where inside 36-39 the gain sits, and whether Q8_0 for the very last layer beats Q4_K/Q6_K.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python; Q8=/home/hikkian/Desktop/AI/bench/rufix/src/Ornith-1.5-35B-Q8_0.gguf
systemctl --user stop ai-gateway; sleep 6
echo "scan4 start $(date +%T)"
ALL=ffn_gate_exps,ffn_up_exps,ffn_down_exps
for v in "l39:39:$ALL:$W/donor-Q4_K_M.gguf" "l38:38:$ALL:$W/donor-Q4_K_M.gguf" "l37-39:37-39:$ALL:$W/donor-Q4_K_M.gguf" "q8l39:39:$ALL:$Q8" "q8l38-39:38-39:$ALL:$Q8"; do
  IFS=: read L R K D <<< "$v"; O=$W/mix-$L.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $D --out $O --layers $R --kinds $K > $W/splice-$L.log 2>&1 || { echo "splice $L failed: $(tail -1 $W/splice-$L.log)"; continue; }
  grep "Итого" $W/splice-$L.log | cut -c1-120
  $W/eval_variant.sh $O $L; rm -f $O
done
echo "scan4 end $(date +%T)"
systemctl --user start ai-gateway
touch $W/SCAN4_DONE
