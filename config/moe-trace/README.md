# Routing profiles for the expert cache (`--moe-cache-profile`)

Format: one row per (position, layer): `pos,layer,id0,id1,...` (decode rows only, produced by `llama-moe-trace`).

| file | model it was recorded on | layers x experts | decode positions per layer | status |
|---|---|---|---|---|
| `tiel-coder-agentic.csv` | Tiel-Coder | 40 x 256 | ~4.4k (recorded earlier) | used by Tiel |
| `occamy-agentic.csv` | Occamy-1.0 APEX | 40 x 256 | 4820 (GEMMACACHE1, 11 requests) | used by Occamy (profile A): agentic turn with 26k prompt +25 % faster, decode -3 to -4 % |
| `gemma-agentic.csv` | Gemma 4 26B-A4B QAT | 30 x 128 | 4927 (GEMMACACHE1, 11 requests) | recorded, not adopted: agentic turn +0.6 % (no win) |

GEMMACACHE1 requests: 8 short Russian coding and general questions, 3 agentic requests (ShuraCode-like system prompt, tool schemas, a step plan, then a JSON call). Previous Occamy profile: git history of this file (commit 0fa3b95 and earlier).

Note: a profile recorded on one model is ranked by frequency and used for any model layers it covers; experts with ids beyond the model's count are ignored. A profile of another model therefore fills the cache with that model's favourites, which is not random but unrelated to the running model's routing.
