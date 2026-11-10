#!/bin/bash
# Layer 39 only, per tensor kind.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python
while [ ! -f $W/SCAN4_DONE ]; do sleep 15; done
systemctl --user stop ai-gateway; sleep 6
echo "scan5 start $(date +%T)"
for v in "l39gu:ffn_gate_exps,ffn_up_exps" "l39dn:ffn_down_exps"; do
  L=${v%%:*}; K=${v#*:}; O=$W/mix-$L.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K_M.gguf --out $O --layers 39 --kinds $K > $W/splice-$L.log 2>&1 || { echo "splice $L failed: $(tail -1 $W/splice-$L.log)"; continue; }
  grep "Итого" $W/splice-$L.log | cut -c1-120
  $W/eval_variant.sh $O $L; rm -f $O
done
echo "scan5 end $(date +%T)"
systemctl --user start ai-gateway
touch $W/SCAN5_DONE
