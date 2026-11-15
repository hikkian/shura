# Review of the hand-written queries (before any measurement)

The 222 queries in `queries.raw.json` were written blind (a Sonnet subagent that saw only `records.json`, not the search code) and read in full by the
session before the first run. Changes made at that review, none of them looking at any arm's results:
- `negative_deleted-05` (Occamy launch profile with turbo3) and `negative_deleted-06` (Vulkan prefill) removed: live records partly cover both topics
  (`board-lanes`/`default-model` mention Occamy, `turbo4-kv` says Vulkan with turbo4 gives garbage), so they were not clean negatives. The group has 4 queries.
- Expected keys widened where a live record is a correct answer too: version-03/04 += `undervolt-off-week`; version-05/06 += `main-model-tiel`;
  version-07/08 += `run-gateway-suite`; version-10 += `mixed-slots-valve`; version-11/12 += `reply-in-russian`.
- The deleted topic "taskset" was not turned into a negative: the live record `mixed-threads` (do not use taskset on the server) answers it.

# LL2b: review of `queries2.raw.json` (new queries, before any measurement)
Written blind by a fresh subagent (saw only `records.json`). Read in full by the session. Removed `negative_deleted-02` (Occamy launch profile) and
`negative_deleted-03` (Vulkan prefill) for the same reason as in LL2: live records partly cover both topics. The group has 2 queries. Expected keys
that list a second live record next to a chain key (`undervolt-off-week`, `run-gateway-suite`) were left as written: both records answer the question.
Typo queries are generated with seed 20261109 (`make_queries.py --raw queries2.raw.json --out queries2.json --seed 20261109 --counts plain_ru=16,plain_en=10,mixed=7,code=7`).
