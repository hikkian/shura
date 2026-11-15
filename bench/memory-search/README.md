# bench/memory-search — LL2: measuring memory search without a model

What it is: a fixed, synthetic memory (about 130 live records in the style of one developer's notes, ru/en/mixed; invented text, no real
personal data) and 250+ queries in groups (plain ru/en, code, mixed, typos, word forms, paraphrases, versions, disputes, cross-language, negatives).
It builds a real `gateway/lifelong_store.py` store, runs the queries against several search arms, and prints the metrics and the verdicts of the
rules written down in `PREREG.md` **before** the first run.

Files
- `PREREG.md` — arms, metrics, decision rules (and the clarifications made before the first run). Do not change the rules after a run.
- `fixtures/records.json` — the synthetic memory. `fixtures/queries.raw.json` — hand-written queries (written blind to the search code).
  `fixtures/REVIEW-NOTES.md` — what was changed in the queries at review. `fixtures/queries.json` — raw + generated typo queries.
- `make_queries.py` — builds `queries.json` (typos from a fixed seed). `run_ll2.py` — the measurement.
- Tests: `tests/test_memory_search_bench.py` (fixtures, metrics, decision rules, arms).

Run (CPU only; databases on tmpfs; no GPU, no model, no network):

    python3 bench/memory-search/make_queries.py
    nice -n 19 python3 bench/memory-search/run_ll2.py --out /dev/shm/ll2-out

The arms: `u61` (production search), `tri` (FTS5 trigram, OR of all trigrams), `both` (sum of the two), and the exploratory `stem` (cut word
endings, prefix match) and `triw` (whole words as substrings). The synthetic memory is a stand-in: real memory is smaller, in the user's own words,
and grows slowly; treat the numbers as a comparison between arms, not as a promise.

LL2b (`PREREG-LL2b.md`, `run_ll2b.py`, `fixtures/queries2*.json`): the `stem` arm on new queries and the budgeted memory index. The store flags it
led to are off by default: `Store(stem=True)` / gateway `memoryStem`, and `Store(index_budget_tokens=1500)` / gateway `memoryIndexBudgetTokens`.
