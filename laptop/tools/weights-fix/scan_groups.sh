#!/bin/bash
# Group scan: base, then expert tensors (gate/up/down) of layers 0-9, 10-19, 20-29, 30-39 raised to the Q4_K_M donor types, one group at a time.
W=/home/hikkian/Desktop/AI/weights-fix; BASE=/home/hikkian/.local/share/shura/models/Ornith-1.5-35B-A3B-GSQ-RCO-3bit-MTP.gguf; PY=/home/hikkian/Desktop/AI/bench/.venv/bin/python
systemctl --user stop ai-gateway; sleep 8
echo "scan start $(date +%T)"
$W/eval_variant.sh $BASE base
for g in 0-9 10-19 20-29 30-39; do
  O=$W/mix-$g.gguf
  $PY $W/splice_gguf.py --base $BASE --donor $W/donor-Q4_K_M.gguf --out $O --layers $g --kinds ffn_gate_exps,ffn_up_exps,ffn_down_exps > $W/splice-$g.log 2>&1 || { echo "splice $g failed"; continue; }
  $W/eval_variant.sh $O g$g
  rm -f $O
done
echo "scan end $(date +%T)"
systemctl --user start ai-gateway
touch $W/SCAN1_DONE
