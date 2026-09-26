# Verification discipline (anti-hallucination)

- Never claim a task is complete, a bug is fixed, or a test passes without actually running it and seeing the real output. If you haven't run something, say "not yet verified" rather than assuming success.
- Never state a library/API behavior, function signature, or file content from memory when you can check it directly (read the file, grep the codebase, or search the web for current docs). Prefer checking over recalling for anything that could be version- or project-specific.
- If genuinely uncertain about a fact, say so explicitly rather than presenting a guess as confirmed. A clearly-flagged guess is far more useful than a confident wrong answer.
- Before reporting a fix as done: re-run the failing test/reproduction and show the actual passing output, not just describe the code change.
- When web search or fetched docs contradict your prior assumption, trust the fresh source over your own memory - training data has a cutoff and this project's dependencies may have moved since.
- Distinguish clearly between "I verified this" and "this should work" in your own wording - don't blur the two.

# File paths in tool calls

- In every tool call (read, edit, write, glob, grep, bash), refer to files by paths RELATIVE to the current project directory (e.g. `src/app.py`, `./test_buggy.py`). Never type out an absolute `/home/...` path yourself — the home directory name is easy to mistype, and a mistyped absolute path is rejected as being outside the project.
- If you need a file's location, get it from a glob/grep/ls result and reuse that path exactly as returned.
