# Which model is better for our agent? A protocol fixed before the first run

**Question.** Of the models people run on a 12 GB GPU for agentic coding, which one solves more real engineering tasks with the
ShuraCode agent, at the same size class (IQ4_XS) and the same settings? Candidates (arms): `tiel` (Tiel-Coder as shipped, terse
mode on), `tiel-no-terse` (the same file, `chat_template_kwargs {"terse": false}`), `base` (Qwen3.6-35B-A3B, unsloth MTP GGUF),
optionally `occamy` (Occamy-1.0).

**What is measured.** A task is a small project with a precise specification (`implement`) or a working module with one injected
bug and a visible failing test (`bugfix`). The outcome is objective: hidden unit tests, copied in after the agent finished.
Nobody judges the code. Tasks are written after the models' training cutoffs, nothing is taken from public benchmarks.

**What is the same for every arm.** The agent (`shuracode run --format json --auto`), its tools, the system rules (a fixed minimal
set, no memory plugin), sampling (temperature 0.6, top-p 0.95), context (32k per slot), limits (15 minutes of wall time per
task, 32k context), the llama-server flags, the hardware, the order of the night (every arm does the same tasks in every round).

**Unit of analysis.** The task *family* (a family is one idea with two variants; the variants are not independent). Repetitions
of one task are averaged first. Arms are compared pairwise on the same tasks.

**Statistics (analyze.py, tested by simulation in simulate.py and power.py).** Paired sign-flip randomisation test (100,000 flips),
95% cluster bootstrap CI, Holm correction over all pairs. **Winner:** one arm significantly better (Holm-adjusted p < 0.05) than
every other arm. **Otherwise a tie** among the arms not significantly worse than the best, broken by tokens per solved task
(median over tasks all tied arms solved). The solved-per-hour figure is reported as well.

**Early stop.** After each round (12 tasks per arm), a Haybittle-Peto look: stop only if a winner has Holm-adjusted p < 0.001
against every other arm. Otherwise run to the end of the task set or the time budget (14 h).

**What this can and cannot say** (from power.py): with about 100 independent units a real gap of ~10-12 points is found with
roughly 80% probability; a gap of 5 points is not. A tie therefore means "no difference larger than about 10 points was
shown", not "the models are equal".

**What it does not measure.** Building a product from nothing to production, long-horizon planning, security, taste. It
measures scoped engineering tasks done through the real agent.
