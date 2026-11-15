# Effort (EFFORT1): the slider, what each level really does, and how it is calibrated

The user picks how hard the model works, the way the effort setting works in Claude Code: a slider with fixed levels, no
automatic mode. Inside a level the model keeps its own judgement (SAGE1: at one level Tiel writes 356 tokens for a bugfix step
and 917 for an implement step). A level is therefore not a phrase in the template but three things at once: the template's
switch, a soft target for the reasoning of one step, and a hard ceiling that ends the reasoning gracefully instead of cutting
it mid-sentence.

Everything here is opt-in. Without `effortBudgets` in `guardian.json` the gateway behaves as before (THINK2): the level only
adds the template's `chat_template_kwargs`.

## 1. The slider

| level | what the user gets | why it exists |
|---|---|---|
| `off` | no thinking: the answer starts at once | simple questions and chat; only models whose template can switch thinking off (`enable_thinking: false`) |
| `low` | a short think, then the answer | quick answers, small edits; the model is told to be brief and the ceiling is low |
| `medium` | the default: the model decides how much to think | the measured behaviour of SAGE1; the budgets only cut the far tail |
| `high` | thinks longer and checks itself | hard tasks; the template asks for careful reasoning, the ceiling is far |
| `max` | as long as it needs, up to the largest ceiling | the hardest tasks; slow by design |

Decisions: five levels, `medium` in the middle and as the default; `off` stays because the product already had it and it is
honest for a model that can do it (Claude Code has no off, but its models cannot answer without thinking at all). The level is
a session switch kept by the gateway (`state/think.flag`), survives a restart, is changed with one command in ShuraCode
(`/effort`, `/think` is the same) and is always visible in the context chip (`effort: medium`). No automatic level: the user
asked for the frontier's model, where the person moves the slider and the model adapts inside it.

## 2. From a level to a request

The gateway (`gateway/user_modes.py`, `apply_think`) turns the level into request fields:

1. `chat_template_kwargs` from `thinkLevels[level]` (THINK2, unchanged): for Tiel's Sharp template `reasoning_effort` low /
   xhigh and `enable_thinking: false`; for Occamy `reasoning_effort` low / high; a level the template lacks runs at
   `thinkFallback[level]` (Occamy: `max` → `high`, `off` → `low`), else at `medium`.
2. With `effortBudgets: true` in `guardian.json`, the level's budget from `effortBudgets[level]`:
   - `reasoning_budget_tokens` = `ceilingTokens`: the server's reasoning-budget sampler (shura.cpp, upstream `--reasoning-budget`)
     counts tokens from the thinking start tag and, at the ceiling, forces `reasoning_budget_message` and the end tag;
   - `reasoning_budget_message` = `message` or the default wrap-up line ("Thinking time is up. I will stop here and answer with
     my best current understanding."), so the transcript stays coherent and the answer follows;
   - `reasoning_budget_soft_tokens` = `targetTokens` (when set): the soft target for the server part below (§4); a server
     without it ignores the field.
   `off` sends no budget (nothing to budget); a level without a budget entry sends none.
3. A client's own choice wins: an OpenAI `reasoning_effort` in the request names the level for that one request
   (`none`/`off` → off, `minimal`/`low` → low, `medium`, `high`, `xhigh`/`max` → max; the field itself still reaches the
   template as before), and a client's `reasoning_budget_tokens` / `thinking_budget_tokens` are kept untouched.

Inside the level the model is free below the target; the ceiling is the only hard limit, and it ends with a sentence of its own.

### Switches

| where | key | default | meaning |
|---|---|---|---|
| guardian.json | `effortBudgets` | `false` | send the level's budget fields; needs a server with the reasoning-budget sampler (shura.cpp on the current upstream; the live binary of 2026-10 does not have it and would ignore the fields). Asking for it selects the new gateway entrypoint. |
| model-launch, per model | `thinkLevels` | absent | level → `chat_template_kwargs` (THINK2); now also `max` |
| model-launch, per model | `thinkFallback` | absent | level the template lacks → the level used |
| model-launch, per model | `effortBudgets` | absent | level → `{targetTokens, ceilingTokens, message}`; `null` = none; target ≤ ceiling |

### API

- `POST /guardian/effort-<level>` (alias of `/guardian/think-<level>`): `off | low | medium | high | max | default`; the answer is the
  snapshot below. `GET /guardian/status` → `think`: `level`, `default`, `effective`, `available`, `fallback`, `levels` (the slider),
  `budgets` (switch on?), `budget` (`{target, ceiling}` of the effective level when budgets are on).
- `/v1/chat/completions`: `reasoning_effort` as above; `reasoning_budget_tokens`, `reasoning_budget_message` pass through.

## 3. Budgets per model (stubs until calibration)

Source: the SAGE1 agent logs (2026-10-26, 56 coding tasks × 2 repeats, `/think medium`), output tokens per step, of which 83 %
are reasoning by characters.

| model | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| Tiel (Sharp, medium) | 246 | 783 | 1620 | 3016 | 7354 | 15727 |
| Sage (own template) | 387 | 1178 | 2359 | 4475 | 10250 | 16384 (the output cap) |

Stubs, reasoning tokens per step (`targetTokens` / `ceilingTokens`); the full blocks for every model, including the
`thinkLevels` each template understands, are in `reports/effort1/effort-budgets.json` and validate against the gateway:

| model | low | medium | high | max |
|---|---|---|---|---|
| Tiel-Coder, Occamy (same family) | 300 / 1500 | 1200 / 6000 | 4000 / 16000 | – / 24000 |
| Sage (×1.5) | 450 / 2200 | 1800 / 9000 | 6000 / 16000 | – / 24000 |
| Gemma 4 26B-A4B (on/off template only) | – / 1200 | – / 5000 | – / 12000 | – / 20000 |

The small model for 6–8 GB cards is chosen in INSTALL1 and gets its levels and budgets there, with the same keys.

How the Tiel numbers were chosen: the medium ceiling (6000) sits between p95 and p99, so it touches about 2 % of the steps the
model takes today and never the typical ones; the low ceiling (1500) is near p90, so low really is short; high's ceiling is above
the largest step ever seen, max's is bounded by the window (24k of a 32k window leaves room for prompt and answer). Targets are a
third to a quarter of the ceilings and are only used by the server part (§4). Gemma's numbers are analogies (no SAGE1 data for
it) and must be replaced by the calibration.

## 4. The server part (shura.cpp): soft target, graceful ceiling

Upstream's reasoning-budget sampler already gives the ceiling with a wrap-up line and realtime `reasoning_end`
(`/v1/chat/completions/control`). What it lacks, and what the rewritten THINK3 card asks for (`reports/effort1/THINK3b.txt`):

1. `reasoning_budget_soft_tokens` (per request): below it the sampler never intervenes.
2. Between the target and the ceiling, ThinkBrake (arXiv 2510.00546): at a sentence boundary inside the thinking block compare the
   logit of the end tag's first token with the top-1; within the margin τ, force the end tag. No training, the model's own
   confidence decides; the paper reports up to −30 % reasoning at the same accuracy.
3. At the ceiling, a grace window: wait for a sentence end (up to `reasoning_budget_grace_tokens`, default 48) before forcing
   the wrap-up line and the end tag, so the cut lands between sentences.
4. Speculative decoding (MTP): the forced token is applied in the verification step, so a draft that disagrees is simply rejected
   from that position; nothing to do beyond a test.

Until the server has 1–3, a level is the template switch plus the hard ceiling with the wrap-up line: already more than a phrase.

## 5. The agent on high and max

Not wired. The cheap and honest candidate is the completion gate (an explicit verification step before the agent declares the task
done, QLEV1 arm G): it is turned on for `high` and `max` only if arm G passes its gate in QLEV1b. Until then the levels change
the model's reasoning only.

## 6. Monotonicity and the calibration plan

Rule: each higher level is not worse in quality and not faster in time than the one below; `low` may lose a little quality for a
real gain in time. Measured, not assumed.

- Tasks: the dev part of RSIT1 (56 coding tasks), the serve mode of the harness (as the product), the QLEV1 base settings.
- Arms, each one difference from the base (`medium` with budgets off): `low`, `medium` with budgets on, `high`, `max`; then the
  server part (soft target + ThinkBrake) as its own arm on `medium`.
- Metrics per arm: resolved and the soft score (paired difference, 95 % bootstrap CI by task), time per task, reasoning tokens
  per step (p50/p90/p99), how often the ceiling fired (the wrap-up line in the transcript), timeouts.
- Gates (DATASET.md §5): `high`/`max` accepted when resolved is not worse than `medium` (lower CI bound > −0.03) and score is not
  worse; `low` accepted when resolved is not worse than −0.03 and time per task is lower; a ceiling that fires in more than 5 % of
  medium steps is raised before the arm is judged.
- Cost: 56 tasks × ~195 s ≈ 3 GPU-hours per arm and repeat; 4 arms × 2 repeats + base ≈ 27 GPU-hours for Tiel; Occamy and Sage
  after it. Runs at night under the usual rules (VRAM watchdog, quiet), as QLEV1b.
- Output: the numbers replace the stubs in `reports/effort1/effort-budgets.json`; then the candidate config.

## 7. Compatibility

- A template without levels (Gemma 4): `thinkLevels` knows only `off`/`medium`, the other levels fall back to
  `medium` and differ only by their ceiling. A model without any of this: requests are untouched.
- A server without the budget sampler ignores the fields; the switch stays off until the live binary is a shura.cpp build with it.
- The legacy gateway entrypoint has no levels; `effortBudgets: true` selects the new one.
- ShuraCode: `/effort` (alias `/think`) with five levels; the chip shows the level; against a gateway from before EFFORT1 the plugin
  falls back to `think-<level>` (no `max` there).

## 8. Not verified

- No GPU run yet: the budgets are stubs, the wrap-up line's effect on answer quality is unmeasured, the server part is a spec.
- Whether the Sharp template's `xhigh` line for both `high` and `max` is the right split (the ceilings differ, the words do not).
- The ShuraCode plugin was not compiled here (no bun/esbuild on this machine): the pure module is tested with node, the TSX was
  reviewed by eye and will be exercised by the engine's self-test on the next `shuracode update`.
