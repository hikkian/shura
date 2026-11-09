#!/bin/bash
# valid answers must pass the hidden test, wrong ones must fail. Usage: ./check_answers.sh
D=$(cd "$(dirname "$0")" && pwd); T=$(dirname "$D")
W=$(mktemp -d); bad=0
run() { rm -rf "$W/c"; mkdir "$W/c"; cp -r "$T/repo/." "$W/c/"; cp "$T/hidden_test.py" "$W/c/"
  (cd "$W/c" && ANSWER_FILE="$1" python3 -W ignore -m unittest hidden_test >/dev/null 2>&1); }
for f in "$D"/answer.txt "$D"/answer_alt_*.txt; do run "$f"; r=$?; echo "valid  $(basename $f): $([ $r = 0 ] && echo PASS || echo 'FAIL (UNEXPECTED)')"; [ $r = 0 ] || bad=1; done
for f in "$D"/wrong_*.txt; do run "$f"; r=$?; echo "wrong  $(basename $f): $([ $r != 0 ] && echo 'FAIL (expected)' || echo 'PASS (UNEXPECTED)')"; [ $r != 0 ] || bad=1; done
rm -rf "$W"; exit $bad
