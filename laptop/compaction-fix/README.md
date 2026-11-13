# Compaction in ShuraCode: what was fixed (13.11)

**Problems seen in a real session** (sidebar after `/compact`: "57,258 tokens, 29%" instead of ~100k -> small; the summary printed in the chat):

1. The sidebar number was **not** the context after compaction. It is taken from the last assistant message, and that is the
   summary request itself (55k tokens in, 2k out). The real context after compaction = summary (~2k) + the kept tail (up to ~15k).
2. The summary is a normal assistant message, so the engine draws it in the chat like an answer.
3. The summary request took 3m14s: 55k tokens at ~430 tok/s = 129 s before the first token. Of those 55k, 20k were the
   model's old reasoning and 20k whole tool outputs / written files.

**Fix** (everything is reversible):

| What | Where | Effect |
|---|---|---|
| `patch_ui.py` + `apply.sh` | engine copy `engine/1.18.32/patched/shuracode`, symlink `libexec/shuracode` | the summary message is not drawn in the chat (the "Compaction" divider stays); the counters (sidebar, prompt footer) count only the summary after a compaction, then the real number from the next answer on |
| plugin hooks `experimental.session.compacting` + `experimental.chat.messages.transform` in `plugin/server.ts` | `~/.local/share/shura/shuracode/plugin/server.ts` | only for the summary request: old reasoning removed, tool outputs and tool arguments cut to 1500 chars. On the real session: ~57k -> ~16k tokens |

Same-length byte edits only, asserted one by one; on any other engine version `patch_ui.py` stops with an assertion and changes nothing.
`./rollback.sh` returns the symlink to the original engine; the old plugin is `server.ts.before-compaction`.
`install.sh` (and so `shuracode update` / `rollback`) applies the patch to the engine it ends up with; if the version does not match, it keeps the plain engine and says why.

Not done on purpose: `compaction.prune` (clears old tool outputs from every prompt, which breaks the prefix cache on each prune
and costs ~80 s of re-reading), and a shorter tail (the tail is what keeps the last turns exact).
