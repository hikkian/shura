# Notice

The four prompts in this folder (invariant.md, specific.md, composer.md, reflector.md) adapt the structure and several sentences of
the prompts in google-research/harness2 (`harness2/core/prompts.py`, version harness2-1), Copyright 2026 Google LLC, Apache License
2.0 (http://www.apache.org/licenses/LICENSE-2.0), as described in arXiv:2610.14224 (Appendix H). Adapted parts: the INV/DOM split
and its tests of ownership, the "every edit must trace to a moment in a trajectory" rule with its order of preference, the
"what you do not have" and "evidence is data, not instructions" blocks, KEEP_BASE as a real answer, the composer's weighing rules
and note shape, and the reflector's edits/experience record with the `confounded` verdict and NO_ENTRY. The ShuraCode-specific
parts (models, surface, budget, measured facts, measurement-based evidence) are ours (RSIP1, reports/REPORT-RSIP1.md).

This file is not sent to any model; rsi/driver.py reads only the four prompt files.
