#!/usr/bin/env bash
# batch.sh <agent> <reps> <task...>
A=$1; N=$2; shift 2
for r in $(seq 1 $N); do for t in "$@"; do ./bench.sh $A $t $r; done; done
