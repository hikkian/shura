#!/usr/bin/env bash
# CODEEVAL2: the JS dependencies of the frontend/backend eval tasks (react, react-dom, express, typescript, playwright-core), installed ONLY from the
# local npm cache (--offline: nothing is downloaded; a package missing from the cache fails the script). Result: vendor_js/node_modules (~60 MB, not in git).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vendor_js && cd vendor_js
[ -f package.json ] || echo '{"name":"ce2-vendor","private":true}' > package.json
for p in react react-dom typescript express playwright-core @types/react; do
  npm install --offline --no-audit --no-fund "$p" >/dev/null
done
node -e "for (const m of ['react','react-dom','express','typescript','playwright-core']) console.log(m, require(m+'/package.json').version)"
