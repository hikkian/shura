#!/usr/bin/env bash
# CI diagnostic: can the fake llama-server be reached on this runner? Prints the answer as annotations.
note() { echo "::notice title=diag::$(printf '%s' "$1" | tr '\n' '\r' | sed 's/\r/%0A/g')"; }
port=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
python tests/fixtures/fake_llama_server.py --port "$port" >/tmp/fake.log 2>&1 &
pid=$!
sleep 4
out="python: $(command -v python) $(python -V 2>&1)
proxy env: $(env | grep -i proxy | tr '\n' ' ')
server alive: $(kill -0 $pid 2>&1 && echo yes || echo no)
fake log: $(head -c 400 /tmp/fake.log)
curl: $(curl -sS -m 5 "http://127.0.0.1:$port/health" 2>&1 | head -c 300)
python urlopen: $(python - "$port" <<'PY'
import sys, urllib.request
try:
    print(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3).status)
except Exception as e:
    print(type(e).__name__, e)
PY
)"
note "$out"
kill $pid 2>/dev/null
