# ideal5 / prompt-v4: rules, what changed and why

Main principle: **KV-cache prefix stability.** Every earlier prompt must be an exact byte prefix of the next turn's
prompt, so llama-server reuses the cache and only the new tokens are processed. Anything that depends on message content
or is appended to a message breaks the prefix from that point on (and a reminder moved to "the last message" breaks it
on every turn). Second principle: no wording written to fix one specific benchmark task.

## Template (ornith-ideal5.jinja / qwen36-ideal5.jinja)

| rule | was (ideal4) | now (ideal5) | why |
|---|---|---|---|
| Language detection block (LANG_BLOCK / LANG_BLOCK4, `<context>` skipping, Cyrillic/Kazakh letter counting) | system block depended on the latest user message: RU_LINE / KZ_LINE / MIX_LINE or nothing | removed entirely | content-dependent system block changes when the user switches language and invalidates the whole cache; ~60 lines of Jinja gone (also fewer engine-compat risks) |
| RU_LINE, KZ_LINE, MIX_LINE (language sentences) | three language-specific sentences, chosen per turn | ONE fixed English sentence, always present, at the end of the system block just before the effort line: "Write everything addressed to the user ... in the language of the user's latest message. Keep code, commands, paths and identifiers as they are ... do not translate them." | model follows the user's latest message itself; the sentence is constant so the system block is byte-identical on all turns (also for English-only chats); `lang_guard.py` in the gateway stays the RU/KZ backstop |
| Reminders RU_REM, KZ_REM, MIX reminder, REMIND3 (medium/high/xhigh/max) | text appended to the LAST user message (`content + "\n\n" + _remind`) or to the last tool result; removed again next turn | removed; nothing is appended to any user/tool/assistant message and no block follows the history | a past message changed by a later turn = prefix break at that message on every turn; effort line at the end of the system block is the only effort text |
| `_remind` variable and REMIND / REMIND3 dicts | present | deleted | dead after the above |
| Effort line position | end of system block | unchanged, but the language sentence now sits right before it, so the effort line is still the last line | A3/A4 |
| medium | no system text, but REMIND3["medium"] appended to last turn | no effort line (native calibration); only the fixed language sentence | A4/A2 |
| EX4 "with one small script extract every code block from the file ... fresh interpreter, one command" | in high / xhigh / max | EX5: "run what you deliver the way the user will: execute every code example and command you put in a file or in the answer exactly as written, in a clean context, and fix whatever fails" | EX4 was added for README-example tasks (README code blocks with missing imports); the principle (run deliverables as shipped) is general, the file-type/tooling details were a task fingerprint |
| IMG4 "transcribe all visible text of the mockup ... item by item" | in high / xhigh / max | IMG5: "ground your work in the inputs: before building, list the concrete facts the inputs state (text, numbers, structure, error messages) and compare your result with that list before answering" | IMG4 was added for the mockup-to-HTML task; same discipline works for any input (logs, specs, errors) |
| "State only what you verified, and say plainly what you could not verify" | max only; high/xhigh "state only what you verified" | all three (high/xhigh/max) + "if the request cannot be fulfilled as stated (missing access or information), say so and what you need instead of simulating it" | general honesty rule; the addition covers requests that cannot be done in the environment |
| "images, logs" in max's "read all the inputs" | "docs, data and inputs (images, logs)" | "docs, data and inputs" | file-kind mention removed |
| edge-case / diff / tests items; adversarial review (max, at most two rounds, stop when a round finds no real defect, review subagent only on request) | as is | unchanged | already general |
| Kept from ideal4 | leading system merge, preserve_thinking (ornith always, qwen36 kwarg default true), `enable_thinking=false` closed empty think block, tolerant string tool arguments, control-token defusing, official tools block + XML tool_call format, mid-conversation system messages | unchanged | A5 |

## System prompt (prompt-v3.txt -> prompt-v4.txt; the production prompt is untouched)

| rule | was (prompt-v3) | now (prompt-v4) | why |
|---|---|---|---|
| Check what you deliver | "code examples in a README or other docs must be complete and runnable as written (imports, real names, no placeholders such as `do_something()`); run each example" | "run every code example and command you put in a file or in the answer exactly as written, in a clean context, and fix whatever fails; examples must be complete (real names from the project, no placeholders)" | added for the README-examples task; README/`do_something()` were the fingerprint, the principle is general and same strength |
| Large files and logs | "(thousands of lines) ... markers (error, exception, timestamps, status codes) with grep ... look for the earliest event, not the loudest one" | "Inputs too large to read whole (very long files, logs, outputs): search them for relevant markers first, then read windows around the hits." | added for a log-analysis task; "earliest event, not the loudest" is task-specific advice and was dropped, the search-then-window principle is kept |
| Screenshot or mockup attached | "write down what is actually in the picture (layout, sections, text, colours, error text) and build or fix from that" | "Ground your work in the inputs you are given (attached images, files, logs, error messages): write down the concrete facts they state, build or fix from that, and compare with that list before answering" | added for the mockup-to-page task; generalised to any input (same text as IMG5 in the template, so the two layers agree) |
| Honesty: cause unknown | long bullet: no access to server/logs -> do not edit, list unconfirmed causes, say what you need | "If the request cannot be fulfilled as stated, or you cannot establish the cause (missing access, logs or data), say so instead of simulating it or presenting a guess as fact: report what you checked, mark causes as unconfirmed, say exactly what you need. Speculative changes only if the user asks." | added for a task where the server/logs were unavailable; same strength, wording no longer tied to server/logs, merged with the template's "cannot be fulfilled" clause |
| Russian proofreading | "Before sending a long text in Russian or Kazakh (README, letter, documentation), re-read for typos, wrong endings and Latin letters inside Cyrillic words" | "Before sending a long text, re-read it once for typos, wrong word forms and stray characters from another alphabet, and fix them." | added after typos in a Russian README; the check is useful for any language, README/Cyrillic were the fingerprint. The separate "Never mix alphabets inside a word" bullet stays |
| Name the source | "name the source (URL) in the reply" | "say which source it came from" | added for a web-research task; the URL requirement was the format fingerprint |

## A6: re-sent assistant turn vs generation prompt + model output
Generation prompt ends `<|im_start|>assistant\n<think>\n`; the model then emits `reasoning\n</think>\n\ncontent[\n\n<tool_call>...]`.
History renders `<|im_start|>assistant\n<think>\n` + reasoning(trim) + `\n</think>\n\n` + content(trim) + `\n\n<tool_call>...</tool_call><|im_end|>\n`
(no `\n\n` before `<tool_call>` when content is empty). These are the same bytes, verified in the check (0 prefix breaks across
user -> tool -> user chains). Residual, unavoidable mismatches (official behaviour, not changed): reasoning/content are `trim`med, so
whitespace the model emitted around them is not reproduced; `enable_thinking=false` turns re-sent with reasoning would render a full
think block (the generation prompt had `<think>\n\n</think>\n\n`). Official-style `ornith.jinja` called WITHOUT `preserve_thinking`
drops the think block of past assistant turns (ideal5 keeps it by A5; with `preserve_thinking=true` the official template
is byte-identical to ideal5 apart from the language sentence).
