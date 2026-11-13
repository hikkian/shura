#!/usr/bin/env python3
"""ShuraCode: UI patch for the OpenCode 1.18.32 engine (same-length byte edits, original file untouched).
1) the compaction summary (assistant message with summary=true) is no longer drawn in the chat; the
   "Compaction" divider line stays;
2) the context counter (sidebar, prompt footer, header) after a compaction counts only the summary itself,
   not the compaction request (input ~55k tokens), which looked like "100k -> 57k".
usage: patch_ui.py SRC DST    (DST is a new file; the launcher symlink is switched separately)"""
import sys, shutil, os, re

src, dst = sys.argv[1], sys.argv[2]
b = open(src, 'rb').read()

def sub(old, new, count=1):
    global b
    old, new = old.encode(), new.encode()
    if len(new) < len(old):
        new += b' ' * (len(old) - len(new))
    assert len(new) == len(old), (old, new)
    n = b.count(old)
    assert n == count, (old[:70], 'found', n, 'expected', count)
    b = b.replace(old, new)

# 1) chat: skip the summary message (role is "assistant" or "user"; "assistant" < "b" < "user").
# One contiguous chunk (user block .. assistant block .. "last" check) so that the total length stays equal:
# two "===" become "==" (-2), the new condition is +2.
a = b'get when(){return C.role==="user"}'
z = b'return T()?.id===C.id'
assert b.count(a) == 1 and b.count(z) == 1
i = b.index(a); j = b.index(z) + len(z)
chunk = b[i:j]
assert 0 < len(chunk) < 3000, len(chunk)
c2 = chunk.replace(a, b'get when(){return C.role=="user"}')
c2 = c2.replace(b'get when(){return C.role==="assistant"}', b'get when(){return C.role<"b"&&!C.summary}')
c2 = c2.replace(z, b'return T()?.id==C.id')
assert len(c2) == len(chunk) and c2 != chunk, (len(c2), len(chunk))
b = b[:i] + c2 + b[j:]

# 2) context counters: three places, the variable name differs
for v in ('W', 'LU', 'c'):
    sub(f'{v}.tokens.input+{v}.tokens.output+{v}.tokens.reasoning+',
        f'({v}.summary?0:{v}.tokens.input)+{v}.tokens.output+')

tmp = dst + '.tmp'  # write beside, then rename: works even while the old copy is running
open(tmp, 'wb').write(b)
os.chmod(tmp, 0o755)
os.replace(tmp, dst)
print('patched ->', dst, len(b))
