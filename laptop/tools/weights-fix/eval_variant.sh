#!/bin/bash
# eval_variant.sh <model.gguf> <label> : (1) typo margin on 24 real typos + code controls (rufix/margin.py, server on 8092, same flags as production),
# (2) perplexity on Russian text (ru_eval.txt, CPU, upstream llama-perplexity). Results appended to results.jsonl. Production gateway must be stopped (RAM).
M=$1; L=$2; W=/home/hikkian/Desktop/AI/weights-fix; X=/home/hikkian/Desktop/AI/bench/rufix
B=/home/hikkian/.local/share/shura/shura.cpp/build-cuda/bin; P=/home/hikkian/.local/share/shura/engines/b11301-vulkan/llama-b11301
LD_LIBRARY_PATH=$B:/usr/local/cuda/lib64 nice -n 10 $B/llama-server --host 127.0.0.1 --port 8092 -m "$M" -c 16384 -np 1 -t 8 -ngl 99 -ncmoe 40 -fa on --jinja --load-mode none > $W/srv-$L.log 2>&1 &
SP=$!
for i in $(seq 900); do curl -s 127.0.0.1:8092/health | grep -q ok && break; kill -0 $SP 2>/dev/null || { echo "server died: $L"; exit 1; }; sleep 2; done
MG=$(PORT=8092 python3 $X/margin.py $L 2>&1 | tail -1)
kill $SP; sleep 10
PPL=$(LD_LIBRARY_PATH=$P nice -n 10 $P/llama-perplexity -m "$M" -f $W/ru_eval.txt -c 2048 --chunks 6 -t 8 -ngl 0 2>&1 | grep -E "Final estimate|PPL" | tail -1)
echo "{\"label\":\"$L\",\"margin\":${MG#ИТОГ },\"ppl\":\"$PPL\"}" | tee -a $W/results.jsonl
