#!/bin/bash
# Refinement inside the best group (layers 30-39): sub-ranges and tensor kinds.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python
systemctl --user stop ai-gateway; sleep 6
echo "scan3 start $(date +%T)"
ALL=ffn_gate_exps,ffn_up_exps,ffn_down_exps
for v in "l38-39:38-39:$ALL" "l36-39:36-39:$ALL" "l30-33:30-33:$ALL" "gu30-39:30-39:ffn_gate_exps,ffn_up_exps" "dn30-39:30-39:ffn_down_exps"; do
  IFS=: read L R K <<< "$v"; O=$W/mix-$L.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K_M.gguf --out $O --layers $R --kinds $K > $W/splice-$L.log 2>&1 || { echo "splice $L failed: $(tail -1 $W/splice-$L.log)"; continue; }
  grep "Итого" $W/splice-$L.log
  $W/eval_variant.sh $O $L; rm -f $O
done
echo "scan3 end $(date +%T)"
systemctl --user start ai-gateway
touch $W/SCAN3_DONE
