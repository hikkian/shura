# Writing a public control task (LL3a)

A control task answers one question: "is the model (with its harness) better than before on a realistic piece of work?". The public layer
is written by us from scratch on the most common kinds of projects (backend/REST, frontend logic, CLI, data processing, parsers, SQL,
automation scripts, tests and refactoring, documentation, security). No third-party code is copied: write the project and the tests yourself.

## Layout (same as tasks_v3, see reports/rsit1/FORMAT.md; an existing example: tasks_v3/py_rle/bugfix and tasks_v3/py_ledgerkit_tax_bug)

```
tasks_control_public/<family>/<kind>/
  task.json   id "<family>-<kind>", family, kind (implement|bugfix|refactor|feature|security|tests-only|docs), lang, difficulty "hard",
              prompt, prompt_lang ("en" or "ru"), test_cmd, timeout_s 900, solution_remove [], visible_cmd (bugfix: reproduction), set "control-public",
              soft "counts", tags [...], source "own", idea "<family without lang prefix>"
  start/      what the agent sees: a small realistic project (3-8 files, packages, a README, some existing tests that pass)
  hidden/     hidden tests, copied by the stand to <work>/hidden_tests/ after the agent has worked; test_cmd runs them
  solution/   files that overlay start/ to give the reference solution (only changed/new files; solution_remove lists files to delete)
  mutants/    m1.json m2.json m3.json = {"file": "<path in the reference tree>", "old": "<text that occurs exactly once>", "new": "<wrong text>"}
```

Standard test_cmd forms: `["python3","-B","-m","unittest","discover","-s","hidden_tests","-t",".","-q"]` (put `__init__.py` into hidden/) or
`["node","--test","hidden_tests/contract.test.js"]`. At least 8 independent test methods, each checks one aspect (a boundary, an error,
the ordinary case, a cross-module effect). Python 3.14 and Node 24, standard library only, no network, deterministic (no wall clock or
random without a seed, no dependence on dict order as meaning), each hidden run under 30 s. Hidden file names must not occur in start/ or in the prompt.

## What makes it a good (hard) control task

- Realistic: the project looks like something a person would really have (handlers + service + storage + config), the prompt is what a
  colleague would write (a bug report or a feature request), not a list of function signatures. The prompt must state the required
  behaviour precisely enough that a correct solution is unambiguous, and must not contain the answer or the hidden test names.
- Needs reading: the change touches 2-4 files; the cause of a bug is not in the file the symptom shows.
- Has traps: at least two edge cases that a quick solution misses (boundary, empty input, unicode, ordering, error path, a second caller
  of the same code, a side effect on existing behaviour). Existing visible tests in start/ must keep passing and some hidden tests check
  that old behaviour is preserved.
- The three mutants are plausible wrong solutions: m1 = the naive/obvious fix, m2 = off-by-one or wrong boundary, m3 = forgets the secondary
  requirement (or breaks an old behaviour). Each must fail the hidden tests; the validator checks it.
- Not a duplicate: look at bench/agentic-eval/splits.json (families) and do not repeat an existing idea.

## Validate (offline, no network, in bwrap)

    cd ~/Projects/shura-integration/bench/agentic-eval
    nice -n 19 python3 control/validate_control.py public --only <family>

Every task must print PASS. Iterate until it does. Do not commit, do not touch other directories, do not run models or GPUs.
