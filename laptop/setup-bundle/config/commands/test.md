---
description: Run the project's tests and fix what fails
agent: build
---
Run this project's test suite and make it pass. $ARGUMENTS

1. Find how tests run here (package.json scripts, pyproject/pytest config, Makefile, Cargo.toml, go.mod,
   README, or your memory of this project) and run the full suite.
2. For each failure: find the root cause and fix the code under test. Change a test only if the test itself
   is wrong, and say so explicitly.
3. Re-run the suite and show the real final result. Do not report success without that output.

If the way to run the tests was not obvious, save it to memory as project knowledge.
