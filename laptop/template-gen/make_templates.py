#!/usr/bin/env python3
"""Generate the ideal chat templates and their single-factor A/B arms from ONE source.

Base = official Qwen3.6 template (XML tool calls, byte-compatible rendering of history),
plus a small number of justified changes (see DESIGN.md):
  1. leading system messages merged (official raises on a second system message)
  2. preserve_thinking default ON (official default is OFF; llama-server passes it anyway)
  3. reasoning_effort guidance appended at the END of the system block (never before tools)
     with 'max' recognised; 'medium' injects nothing
  4. optional `terse` block (Sharp's exact text), default OFF, switchable via chat_template_kwargs
  5. string tool arguments tolerated (official crashes on OpenAI-style JSON strings)
  6. no 'No user query found' crash; mid-conversation system messages rendered instead of raising
  7. control tokens (<|im_start|>, <|im_end|>, vision markers) inside user/tool text are defused with a space
Six effort levels: fast = thinking off, low / high / xhigh / max = one line each, medium = nothing.
Everything else (tools block text, tool_call/tool_response rendering, generation prompt) is the
official text verbatim so llama.cpp's qwen3-coder parser and the model's training format match.

Usage: make_templates.py [OUTDIR]
"""
import sys
from pathlib import Path

# ---- effort guidance wordings -------------------------------------------------------------
# Six levels: fast = thinking off (no text), medium = no text (the model's native calibration; renders byte-identically
# to the official template), the other four get ONE line at the tail of the system block. The line is read on every
# agent step, so each one says how to scale thinking to the step (a routine read/test step never needs a long thought)
# and is written for any task kind: code, design, explanation, prose, research.
IDEAL = {
    "low": "Reasoning effort: low. Think briefly and act: decide the next step in a few sentences, do not re-derive what you already know, and do not plan beyond that step.",
    "high": "Reasoning effort: high. Before acting, make sure you understand what the user actually wants and why; consider a couple of approaches (to the code, the design, the wording or the explanation) and pick the better one; think through edge cases and how you will check the result. Routine steps (reading, running tests) still need only a short thought.",
    "xhigh": "Reasoning effort: very high. Take the time to think it through: pin down the goal and the constraints, explore several genuinely different approaches and their trade-offs, look for a better idea than the first one, plan the work, anticipate what could go wrong or be missed, and decide how you will verify the result. Spend the effort where decisions are made; routine steps still need only a short thought.",
    "max": "Reasoning effort: maximum; quality matters more than time. Think deeply and broadly: fully understand the goal and its context, generate and compare several ideas including non-obvious ones, choose the best and justify the choice to yourself, plan carefully, address root causes rather than symptoms, and critically review your own conclusions before acting and before answering. Spend the effort where decisions are made; routine steps still need only a short thought.",
}
# effort v2: higher levels change what the agent DOES (verification depth), not only how long it thinks.
# The full line sits at the tail of the system block; a short reminder is repeated right before each generation
# (inside the last user / tool-result turn), because after dozens of tool steps the system line is far away.
IDEAL2 = dict(IDEAL)
IDEAL2.update({
    "high": "Reasoning effort: high. Correctness matters more than speed. At the start, pin down exactly what is asked (every requirement, explicit and implied) and how the result will be checked, and plan. Routine steps (reading, running a command) need only a short thought. Before your final answer, run a verification pass: re-read the user's request and check each requirement against what you actually did; re-run the tests or the code; read your diff for mistakes and leftovers; run every code example and command you wrote in files or in the answer; state only what you verified.",
    "xhigh": "Reasoning effort: very high. Correctness matters much more than speed. At the start, pin down every requirement (explicit and implied), the constraints and how the result will be checked; compare at least two genuinely different approaches and choose deliberately. Read more code instead of assuming. Routine steps need only a short thought. Before your final answer, run a verification pass: re-read the request and check each requirement; think of the edge cases the task implies (empty or invalid input, errors, boundaries, unusual values) and test them with a quick command or test; re-run the tests; read your diff; run every example and command you wrote; state only what you verified.",
    "max": "Reasoning effort: maximum; quality matters more than time. At the start, pin down every requirement (explicit and implied), the constraints and how the result will be checked; compare several approaches, including non-obvious ones, and choose deliberately; fix root causes, not symptoms. Read more code instead of assuming. Routine steps need only a short thought. Before your final answer: (1) verification pass: re-read the request and check each requirement, test the edge cases the task implies, re-run the tests, read your diff, run every example and command you wrote; (2) independent review: if you changed files, use the task tool to start a general subagent, give it the user's request and the list of changed files, and ask it to critically review the changes for bugs, missed requirements, broken examples and unverified claims, without editing files; fix every valid finding and re-run the checks. State only what you verified.",
})
REMIND = {
    "high": "[effort: high] Before the final answer: re-check every requirement, re-run tests/code, read your diff, run the examples you wrote.",
    "xhigh": "[effort: xhigh] Before the final answer: re-check every requirement, test implied edge cases, re-run tests/code, read your diff, run the examples you wrote.",
    "max": "[effort: max] Before the final answer: full verification pass (requirements, edge cases, tests, diff, examples), then an independent review by a subagent (task tool) of your changes; fix its valid findings.",
}
# effort v4: same structure as v2/v3 plus two task-agnostic checks that the v3 runs showed missing:
# examples were re-typed into ad-hoc commands instead of being run as written in the file (README examples with
# missing imports / undefined names), and text from images/mockups was not transcribed and compared item by item.
EX4 = ("run every code example and command you wrote exactly as written: with one small script, extract every code block "
       "from the file itself (do not retype them) and run each block on its own in a fresh interpreter, in a single command; "
       "every example must be self-contained (its own imports, defined variables, no placeholders) and finish without errors; "
       "after a fix, re-run the same script instead of checking examples one by one")
IMG4 = ("if the task gives an image or mockup, transcribe all of its visible text (headings, every list item, labels, "
        "numbers) exactly before building, and compare your result with that transcription item by item before answering")
IDEAL3 = dict(IDEAL2)
IDEAL3.update({
    "high": "Reasoning effort: high. Correctness matters more than speed. At the start, pin down exactly what is asked (every requirement, explicit and implied) and how the result will be checked, and plan. Routine steps (reading, running a command) need only a short thought. Before your final answer, run a verification pass: re-read the user's request and check each requirement against what you actually did; re-run the tests or the code; read your diff for mistakes and leftovers; " + EX4 + "; " + IMG4 + "; state only what you verified.",
    "xhigh": "Reasoning effort: very high. Correctness matters much more than speed. At the start, pin down every requirement (explicit and implied), the constraints and how the result will be checked; compare at least two genuinely different approaches and choose deliberately. Read more code instead of assuming. Routine steps need only a short thought. Before your final answer, run a verification pass: re-read the request and check each requirement; think of the edge cases the task implies (empty or invalid input, errors, boundaries, unusual values) and test them with a quick command or test; re-run the tests; read your diff; " + EX4 + "; " + IMG4 + "; state only what you verified.",
    "max": ("Reasoning effort: maximum; quality matters more than time, and you have a large thinking budget: use it. "
            "Investigate before deciding: read all the code, docs, data and inputs (images, logs) the task touches instead of assuming; "
            "write down every requirement, explicit and implied, and how each one will be verified. Generate several approaches, "
            "including non-obvious ones, weigh their trade-offs and choose deliberately; fix root causes, not symptoms. "
            "At every decision point think it through fully; treat any surprising output as something to explain, never to ignore. "
            "Routine steps (reading a file, running a command) still need only a short thought. Before your final answer: "
            "(1) verification pass: re-read the request and check each requirement, test the edge cases the task implies, "
            "re-run the tests, read your diff; " + EX4 + "; " + IMG4 + "; "
            "(2) adversarial self-review: assume your work still contains at least one bug, broken example or missed requirement "
            "and hunt for it like a strict reviewer: compare the result with the request word by word, try inputs designed to "
            "break it, re-read every file you changed in full (not only the diff); fix what you find and repeat the checks "
            "until a full pass finds nothing. Start a review subagent only if the user asks for one. "
            "State only what you verified, and say plainly what you could not verify."),
})
REMIND3 = {
    "medium": "Before the final answer: run the tests and every code example you wrote, exactly as written in the file.",
    "high": "[effort: high] Before the final answer: re-check every requirement, re-run tests/code, read your diff, run every example exactly as written in the file (self-contained, no errors), compare with the mockup if there is one.",
    "xhigh": "[effort: xhigh] Before the final answer: re-check every requirement, test implied edge cases, re-run tests/code, read your diff, run every example exactly as written in the file (self-contained, no errors), compare with the mockup if there is one.",
    "max": "[effort: max] Before the final answer: full verification pass (requirements, edge cases, tests, diff, every example run exactly as written in the file, mockup comparison), then an adversarial self-review: assume a bug or missed requirement remains and hunt for it (request word by word, breaking inputs, every changed file re-read in full); fix and repeat until a full pass finds nothing.",
}
# IsValorum "stock" wording (kept for the wording arm); 'max' is mapped onto its high line
STOCK = {
    "low": "[Reasoning Guidance: Keep your internal reasoning brief, concise, and focused strictly on the immediate problem-solving steps without over-analyzing.]",
    "medium": "[Reasoning Guidance: Maintain a balanced, structured reasoning process, addressing the core problem with standard analytical depth before answering.]",
    "high": "[Reasoning Guidance: Reason deeply, thoroughly, and systematically. Formulate an upfront implementation plan, explore edge cases, and verify logic before concluding.]",
    "max": "[Reasoning Guidance: Reason deeply, thoroughly, and systematically. Formulate an upfront implementation plan, explore edge cases, and verify logic before concluding.]",
}

TERSE_LEAD_THINK = "Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable."
TERSE_LEAD_NOTHINK = "Answer directly and concisely. Give the answer with only what it needs to be correct and usable."
TERSE_CORE = (
    "Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; or repeat a point you've already made.\n"
    "Always: keep essential steps, caveats, uncertainties, and specifics \u2014 never drop correctness or a needed warning for brevity. Keep the final answer lean. Use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why \u2014 never omit uncertainty for the sake of brevity.\n"
    "If a user request is genuinely ambiguous, ask a sharp question, don't guess."
)

# Official Qwen3.6 tools block text (verbatim from Qwen/Qwen3.6-35B-A3B chat_template.jinja)
TOOLS_INTRO = "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>"
TOOLS_OUTRO = (
    "\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n"
    "<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n"
    "<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n"
    "</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n"
    "- Required parameters MUST be specified\\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n"
    "</IMPORTANT>"
)


def j(s: str) -> str:
    """Quote a Python string as a single-quoted Jinja literal with \\n escapes (what minijinja/llama.cpp parse)."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"

# effort v3: the answer drifted into English on Russian requests (xhigh 10/20 on the hard subset): after long English
# tool output and English reasoning, the nearest instruction was English. The language of the last real user query is
# detected (Kazakh-specific letters, then common Cyrillic letters) and named explicitly in the per-turn reminder.
CYL = "['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']"
KZL = "['ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і', 'Ә', 'Ғ', 'Қ', 'Ң', 'Ө', 'Ұ', 'Ү', 'Һ', 'І']"
LAL = "['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']"
LANG_BLOCK = (
    "{#- ---- language anchor (effort v3) ----\n"
    "    Decided from the user's own words: the first line of prose (before any ``` block) of the latest real user message,\n"
    "    skipping identifier-like words (paths, file.ext, snake_case, calls, ACRONYMS). A message with fewer than 8 letters\n"
    "    (\"ok\", \"да\") defers to the previous user message. Starts in Cyrillic and >= 50% Cyrillic letters, or >= 90%\n"
    "    Cyrillic -> Russian (Kazakh if Kazakh-specific letters occur); no Cyrillic -> nothing is added (English requests\n"
    "    render exactly as before); otherwise -> a neutral instruction. Letters are counted with split(), because the\n"
    "    llama.cpp Jinja engine cannot iterate over a string. -#}\n"
    "{%- set _CY = CY_LIST %}\n{%- set _KZ = KZ_LIST %}\n{%- set _LA = LA_LIST %}\n"
    """{%- set _lang = namespace(v='', done=false) %}
{%- for message in _msgs[::-1] %}
    {%- if not _lang.done and message.role == 'user' %}
        {%- set _raw = render_content(message.content, false) | trim %}
        {%- if not (_raw.startswith('<tool_response>') and _raw.endswith('</tool_response>')) %}
            {%- set _line = ((_raw.split('```')[0] | trim).split('\\n')[0])[:300] %}
            {%- set _k = namespace(t='', first='') %}
            {%- for w in _line.split(' ') %}
                {%- if '_' not in w and '.' not in w and '/' not in w and '(' not in w and '=' not in w and '`' not in w and not (w == w | upper and w != w | lower) %}
                    {%- set _k.t = _k.t ~ ' ' ~ w %}
                    {%- if not _k.first %}
                        {%- set _f = namespace(c=false, l=false) %}
                        {%- for ch in _CY + _KZ %}{%- if ch in w %}{%- set _f.c = true %}{%- endif %}{%- endfor %}
                        {%- for ch in _LA %}{%- if ch in w %}{%- set _f.l = true %}{%- endif %}{%- endfor %}
                        {%- if _f.c and not _f.l %}{%- set _k.first = 'c' %}{%- elif _f.l and not _f.c %}{%- set _k.first = 'l' %}{%- endif %}
                    {%- endif %}
                {%- endif %}
            {%- endfor %}
            {%- set _n = namespace(c=0, l=0, k=0) %}
            {%- for ch in _CY %}{%- set _n.c = _n.c + (_k.t.split(ch) | length) - 1 %}{%- endfor %}
            {%- for ch in _KZ %}{%- set _n.k = _n.k + (_k.t.split(ch) | length) - 1 %}{%- endfor %}
            {%- for ch in _LA %}{%- set _n.l = _n.l + (_k.t.split(ch) | length) - 1 %}{%- endfor %}
            {%- set _n.c = _n.c + _n.k %}
            {%- if _n.c + _n.l >= 8 %}
                {%- set _lang.done = true %}
                {%- if (_k.first == 'c' and _n.c * 2 >= _n.c + _n.l) or _n.c * 10 >= (_n.c + _n.l) * 9 %}
                    {%- set _lang.v = 'Kazakh' if _n.k > 0 else 'Russian' %}
                {%- elif _n.c > 0 %}
                    {%- set _lang.v = 'mixed' %}
                {%- endif %}
            {%- endif %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if _lang.v == 'mixed' %}
    {%- set _remind = ((_remind ~ ' ') if _remind else '') ~ "Write the final answer in the language of the user's own words in their last message, not the language of code, logs or tool output." %}
{%- elif _lang.v %}
    {%- set _remind = ((_remind ~ ' ') if _remind else '') ~ 'Write the final answer to the user in ' ~ _lang.v ~ ' (the language the user writes in).' %}
{%- endif %}
"""
).replace("CY_LIST", CYL).replace("KZ_LIST", KZL).replace("LA_LIST", LAL)

RU_LINE = "Язык общения: русский. Всё, что ты пишешь пользователю, — короткие пояснения между шагами и итоговый ответ — пиши на русском языке; код, команды, пути и идентификаторы оставляй как есть, а устоявшиеся технические термины, названия и значения параметров (API, commit, endpoint, high, xhigh) пиши так, как их пишут русскоязычные разработчики, не переводи их искусственно."
KZ_LINE = "Қарым-қатынас тілі: қазақ тілі. Пайдаланушыға жазатын барлық мәтінді — қадамдар арасындағы қысқа түсініктемелерді және қорытынды жауапты — қазақ тілінде жаз; код, командалар, жолдар мен идентификаторларды өзгертпе, ал қалыптасқан техникалық терминдерді, атауларды және параметр мәндерін (API, commit, endpoint, high, xhigh) жасанды түрде аудармай, әзірлеушілер жазатындай қалдыр."
RU_REM = "Итоговое сообщение пользователю пиши целиком на русском языке, с первой строки."
KZ_REM = "Қорытынды хабарламаны пайдаланушыға бірінші жолдан бастап толығымен қазақ тілінде жаз."
MIX_LINE = "Write everything addressed to the user (short notes between steps and the final answer) in the language of the user's own words in their latest message, not the language of code, logs or tool output."
_tail3 = LANG_BLOCK[LANG_BLOCK.index("{%- if _lang.v == 'mixed' %}"):]
_RAW_LINE = "        {%- set _raw = render_content(message.content, false) | trim %}\n"
assert _RAW_LINE in LANG_BLOCK
# ShuraCode/opencode put an English <context>…</context> environment block in front of the user's words: skip it.
_RAW_CTX = _RAW_LINE + ("        {%- if _raw.startswith('<context>') and '</context>' in _raw %}"
                        "{%- set _raw = _raw.split('</context>')[-1] | trim %}{%- endif %}\n")
LANG_BLOCK4 = LANG_BLOCK.replace(_RAW_LINE, _RAW_CTX).replace("(effort v3)", "(effort v4)").replace(_tail3, (
    "{%- set _lline = '' %}\n"
    "{%- if _lang.v == 'mixed' %}\n"
    "    {%- set _lline = " + j(MIX_LINE) + " %}\n"
    "    {%- set _remind = ((_remind ~ ' ') if _remind else '') ~ " + j(MIX_LINE) + " %}\n"
    "{%- elif _lang.v == 'Kazakh' %}\n"
    "    {%- set _lline = " + j(KZ_LINE) + " %}\n"
    "    {%- set _remind = ((_remind ~ ' ') if _remind else '') ~ " + j(KZ_REM) + " %}\n"
    "{%- elif _lang.v == 'Russian' %}\n"
    "    {%- set _lline = " + j(RU_LINE) + " %}\n"
    "    {%- set _remind = ((_remind ~ ' ') if _remind else '') ~ " + j(RU_REM) + " %}\n"
    "{%- endif %}\n"
    "{%- if _lline %}\n"
    "    {%- set sys.text = (sys.text ~ '\\n\\n' ~ _lline) if sys.text else _lline %}\n"
    "{%- endif %}\n"))


def build(*, name: str, effort: str, terse_default: bool, preserve: str, identity: str = "") -> str:
    """effort: 'ideal' | 'stock' | 'none'; preserve: 'kwarg' | 'always' | 'never'."""
    w = IDEAL3 if effort == "ideal4" else IDEAL2 if effort in ("ideal2", "ideal3") else IDEAL if effort == "ideal" else STOCK
    rem = REMIND3 if effort == "ideal4" else REMIND if effort in ("ideal2", "ideal3") else {}
    if effort == "none":
        effort_block = "{%- set _guidance = '' %}\n{%- set _remind = '' %}\n"
    else:
        medium = f"    {{%- elif _effort == 'medium' %}}{{%- set _guidance = {j(w['medium'])} %}}\n" if "medium" in w else ""
        if not medium and rem.get("medium"):  # v4: medium keeps the native calibration (no system line), only the reminder
            medium = f"    {{%- elif _effort == 'medium' %}}{{%- set _remind = {j(rem['medium'])} %}}\n"
        effort_block = (
            "{%- set _guidance = '' %}\n{%- set _remind = '' %}\n"
            "{%- if _thinking %}\n"
            f"    {{%- if _effort in ['low', 'minimal'] %}}{{%- set _guidance = {j(w['low'])} %}}\n"
            f"{medium}"
            f"    {{%- elif _effort == 'high' %}}{{%- set _guidance = {j(w['high'])} %}}{{%- set _remind = {j(rem.get('high', ''))} %}}\n"
            f"    {{%- elif _effort == 'xhigh' %}}{{%- set _guidance = {j(w.get('xhigh', w['high']))} %}}{{%- set _remind = {j(rem.get('xhigh', ''))} %}}\n"
            f"    {{%- elif _effort in ['max', 'ultracode', 'extreme'] %}}{{%- set _guidance = {j(w['max'])} %}}{{%- set _remind = {j(rem.get('max', ''))} %}}\n"
            "    {%- endif %}\n"
            "{%- endif %}\n"
        )
    if preserve == "kwarg":
        preserve_expr = "(preserve_thinking if (preserve_thinking is defined and preserve_thinking is not none) else true)"
    elif preserve == "always":
        preserve_expr = "true"
    else:
        preserve_expr = "false"
    lead_think = (identity + " " if identity else "") + TERSE_LEAD_THINK
    lead_nothink = (identity + " " if identity else "") + TERSE_LEAD_NOTHINK

    lang_block = LANG_BLOCK if effort == "ideal3" else ""
    lang_block4 = LANG_BLOCK4 if effort == "ideal4" else ""
    t = f"""{{#- {name}: official Qwen3.6 chat template + merged leading system messages, preserve_thinking default {preserve},
    reasoning_effort guidance ({effort}) at the tail of the system block, optional `terse` block (default {'on' if terse_default else 'off'}),
    tolerant tool-argument rendering. Tool-call format is the official XML one (llama.cpp uses its qwen3-coder parser).
    Generated by make_templates.py; edit that file, not this one. -#}}
{{%- set template_version = {j(name)} %}}
{{%- set image_count = namespace(value=0) %}}
{{%- set video_count = namespace(value=0) %}}
{{#- Control tokens inside user/tool text (web pages, logs, pasted chats) would be tokenised as specials by the server
    (parse_special) and could close or forge a turn; a space inside the marker keeps them as plain text. -#}}
{{%- macro clean(t) -%}}
    {{{{- (t | string) | replace('<|im_start|>', '<|im_start |>') | replace('<|im_end|>', '<|im_end |>') | replace('<|vision_start|>', '<|vision_start |>') | replace('<|vision_end|>', '<|vision_end |>') | replace('<|image_pad|>', '<|image_pad |>') | replace('<|video_pad|>', '<|video_pad |>') -}}}}
{{%- endmacro -%}}
{{%- macro render_content(content, do_vision_count, is_system_content=false) %}}
    {{%- if content is string %}}
        {{{{- clean(content) }}}}
    {{%- elif content is iterable and content is not mapping %}}
        {{%- for item in content %}}
            {{%- if item is string %}}
                {{{{- clean(item) }}}}
            {{%- elif 'image' in item or 'image_url' in item or (item.type is defined and item.type == 'image') %}}
                {{%- if is_system_content %}}
                    {{{{- raise_exception('System message cannot contain images.') }}}}
                {{%- endif %}}
                {{%- if do_vision_count %}}
                    {{%- set image_count.value = image_count.value + 1 %}}
                {{%- endif %}}
                {{%- if add_vision_id is defined and add_vision_id %}}
                    {{{{- 'Picture ' ~ image_count.value ~ ': ' }}}}
                {{%- endif %}}
                {{{{- '<|vision_start|><|image_pad|><|vision_end|>' }}}}
            {{%- elif 'video' in item or 'video_url' in item or (item.type is defined and item.type == 'video') %}}
                {{%- if is_system_content %}}
                    {{{{- raise_exception('System message cannot contain videos.') }}}}
                {{%- endif %}}
                {{%- if do_vision_count %}}
                    {{%- set video_count.value = video_count.value + 1 %}}
                {{%- endif %}}
                {{%- if add_vision_id is defined and add_vision_id %}}
                    {{{{- 'Video ' ~ video_count.value ~ ': ' }}}}
                {{%- endif %}}
                {{{{- '<|vision_start|><|video_pad|><|vision_end|>' }}}}
            {{%- elif 'text' in item %}}
                {{{{- clean(item.text) }}}}
            {{%- else %}}
                {{{{- raise_exception('Unexpected item type in content.') }}}}
            {{%- endif %}}
        {{%- endfor %}}
    {{%- elif content is none or content is undefined %}}
        {{{{- '' }}}}
    {{%- else %}}
        {{{{- raise_exception('Unexpected content type.') }}}}
    {{%- endif %}}
{{%- endmacro %}}
{{%- if not messages %}}
    {{{{- raise_exception('No messages provided.') }}}}
{{%- endif %}}
{{#- ---- mode: thinking on/off, effort, preserve, terse ---- -#}}
{{%- set _effort = (reasoning_effort | string | lower) if (reasoning_effort is defined and reasoning_effort is not none) else '' %}}
{{%- set _thinking = enable_thinking if (enable_thinking is defined and enable_thinking is not none) else true %}}
{{%- if _effort in ['none', 'off'] %}}
    {{%- set _thinking = false %}}
{{%- endif %}}
{effort_block}{{%- set _preserve = {preserve_expr} %}}
{{%- set _terse_on = terse if (terse is defined and terse is not none) else {'true' if terse_default else 'false'} %}}
{{#- ---- leading system messages (merged, joined by blank lines) ---- -#}}
{{%- set head = namespace(n=0, done=false) %}}
{{%- for m in messages %}}
    {{%- if m.role == 'system' and not head.done %}}
        {{%- set head.n = head.n + 1 %}}
    {{%- else %}}
        {{%- set head.done = true %}}
    {{%- endif %}}
{{%- endfor %}}
{{%- set sys = namespace(text='') %}}
{{%- for m in messages[:head.n] %}}
    {{%- set _part = render_content(m.content, false, true) | trim %}}
    {{%- if _part %}}
        {{%- set sys.text = (sys.text ~ '\\n\\n' ~ _part) if sys.text else _part %}}
    {{%- endif %}}
{{%- endfor %}}
{{%- if _terse_on %}}
    {{%- set _terse = ({j(lead_think)} if _thinking else {j(lead_nothink)}) ~ '\\n' ~ {j(TERSE_CORE)} %}}
    {{%- set sys.text = (sys.text ~ '\\n\\n' ~ _terse) if sys.text else _terse %}}
{{%- endif %}}
{{%- if _guidance %}}
    {{%- set sys.text = (sys.text ~ '\\n\\n' ~ _guidance) if sys.text else _guidance %}}
{{%- endif %}}
{{%- set _msgs = messages[head.n:] %}}
{lang_block4}{{%- if tools is defined and tools and tools is iterable and tools is not mapping %}}
    {{{{- '<|im_start|>system\\n' }}}}
    {{{{- "{TOOLS_INTRO}" }}}}
    {{%- for tool in tools %}}
        {{{{- "\\n" }}}}
        {{{{- tool | tojson }}}}
    {{%- endfor %}}
    {{{{- "\\n</tools>" }}}}
    {{{{- '{TOOLS_OUTRO}' }}}}
    {{%- if sys.text %}}
        {{{{- '\\n\\n' + sys.text }}}}
    {{%- endif %}}
    {{{{- '<|im_end|>\\n' }}}}
{{%- elif sys.text %}}
    {{{{- '<|im_start|>system\\n' + sys.text + '<|im_end|>\\n' }}}}
{{%- endif %}}
{{#- ---- index of the last real user query (only matters when not preserving thinking) ---- -#}}
{{%- set ns = namespace(multi_step_tool=true, last_query_index=_msgs|length - 1) %}}
{{%- for message in _msgs[::-1] %}}
    {{%- set index = (_msgs|length - 1) - loop.index0 %}}
    {{%- if ns.multi_step_tool and message.role == "user" %}}
        {{%- set content = render_content(message.content, false)|trim %}}
        {{%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}}
            {{%- set ns.multi_step_tool = false %}}
            {{%- set ns.last_query_index = index %}}
        {{%- endif %}}
    {{%- endif %}}
{{%- endfor %}}
{{%- if ns.multi_step_tool %}}
    {{%- set ns.last_query_index = -1 %}}
{{%- endif %}}
{lang_block}{{#- ---- conversation ---- -#}}
{{%- for message in _msgs %}}
    {{%- set content = render_content(message.content, true)|trim %}}
    {{%- if message.role == "system" %}}
        {{{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}}}
    {{%- elif message.role == "user" %}}
        {{%- if loop.last and _remind %}}
            {{{{- '<|im_start|>user\\n' + content + '\\n\\n' + _remind + '<|im_end|>\\n' }}}}
        {{%- else %}}
            {{{{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}}}
        {{%- endif %}}
    {{%- elif message.role == "assistant" %}}
        {{%- set reasoning_content = '' %}}
        {{%- if message.reasoning_content is defined and message.reasoning_content is string %}}
            {{%- set reasoning_content = message.reasoning_content %}}
        {{%- elif message.reasoning is defined and message.reasoning is string %}}
            {{%- set reasoning_content = message.reasoning %}}
        {{%- elif '</think>' in content %}}
            {{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}}
            {{%- set content = content.split('</think>')[-1].lstrip('\\n') %}}
        {{%- endif %}}
        {{%- set reasoning_content = reasoning_content|trim %}}
        {{%- if _preserve or (loop.index0 > ns.last_query_index) %}}
            {{{{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}}}
        {{%- else %}}
            {{{{- '<|im_start|>assistant\\n' + content }}}}
        {{%- endif %}}
        {{%- if message.tool_calls is defined and message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}}
            {{%- for tool_call in message.tool_calls %}}
                {{%- if tool_call.function is defined and tool_call.function is not none %}}
                    {{%- set tool_call = tool_call.function %}}
                {{%- endif %}}
                {{%- if loop.first %}}
                    {{%- if content|trim %}}
                        {{{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}}}
                    {{%- else %}}
                        {{{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}}}
                    {{%- endif %}}
                {{%- else %}}
                    {{{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}}}
                {{%- endif %}}
                {{%- if tool_call.arguments is defined and tool_call.arguments is not none %}}
                    {{%- if tool_call.arguments is mapping %}}
                        {{%- for args_name, args_value in tool_call.arguments|items %}}
                            {{{{- '<parameter=' + args_name + '>\\n' }}}}
                            {{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}}
                            {{{{- args_value }}}}
                            {{{{- '\\n</parameter>\\n' }}}}
                        {{%- endfor %}}
                    {{%- elif tool_call.arguments is string and tool_call.arguments|trim %}}
                        {{{{- '<parameter=arguments>\\n' + tool_call.arguments + '\\n</parameter>\\n' }}}}
                    {{%- endif %}}
                {{%- endif %}}
                {{{{- '</function>\\n</tool_call>' }}}}
            {{%- endfor %}}
        {{%- endif %}}
        {{{{- '<|im_end|>\\n' }}}}
    {{%- elif message.role == "tool" %}}
        {{%- if loop.previtem and loop.previtem.role != "tool" %}}
            {{{{- '<|im_start|>user' }}}}
        {{%- endif %}}
        {{{{- '\\n<tool_response>\\n' }}}}
        {{{{- content }}}}
        {{{{- '\\n</tool_response>' }}}}
        {{%- if not loop.last and loop.nextitem.role != "tool" %}}
            {{{{- '<|im_end|>\\n' }}}}
        {{%- elif loop.last %}}
            {{%- if _remind %}}
                {{{{- '\\n' + _remind }}}}
            {{%- endif %}}
            {{{{- '<|im_end|>\\n' }}}}
        {{%- endif %}}
    {{%- else %}}
        {{{{- raise_exception('Unexpected message role.') }}}}
    {{%- endif %}}
{{%- endfor %}}
{{%- if add_generation_prompt %}}
    {{{{- '<|im_start|>assistant\\n' }}}}
    {{%- if not _thinking %}}
        {{{{- '<think>\\n\\n</think>\\n\\n' }}}}
    {{%- else %}}
        {{{{- '<think>\\n' }}}}
    {{%- endif %}}
{{%- endif %}}
"""
    return t


VARIANTS = {
    # ---- Qwen3.6 ----
    "qwen36-ideal":               dict(effort="ideal", terse_default=False, preserve="kwarg"),
    "qwen36-ideal-terse":         dict(effort="ideal", terse_default=True,  preserve="kwarg"),
    "qwen36-ideal-nopreserve":    dict(effort="ideal", terse_default=False, preserve="never"),
    "qwen36-ideal-effort-stock":  dict(effort="stock", terse_default=False, preserve="kwarg"),
    "qwen36-ideal-effort-none":   dict(effort="none",  terse_default=False, preserve="kwarg"),
    # ---- Ornith-1.5 (its own template always renders past thinking -> preserve forced) ----
    "ornith-ideal":               dict(effort="ideal", terse_default=False, preserve="always"),
    "ornith-ideal2":              dict(effort="ideal2", terse_default=False, preserve="always"),
    "qwen36-ideal2":              dict(effort="ideal2", terse_default=False, preserve="kwarg"),
    "ornith-ideal3":              dict(effort="ideal3", terse_default=False, preserve="always"),
    "qwen36-ideal3":              dict(effort="ideal3", terse_default=False, preserve="kwarg"),
    "ornith-ideal4":              dict(effort="ideal4", terse_default=False, preserve="always"),
    "qwen36-ideal4":              dict(effort="ideal4", terse_default=False, preserve="kwarg"),
    "ornith-ideal-terse":         dict(effort="ideal", terse_default=True,  preserve="always",
                                       identity="You are Tiel-Coder, a variant of Ornith-1.5-35B-A3B."),
    "ornith-ideal-effort-none":   dict(effort="none",  terse_default=False, preserve="always"),
}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    for name, kw in VARIANTS.items():
        text = build(name=name, **kw)
        (out / f"{name}.jinja").write_text(text, encoding="utf-8", newline="\n")
        print("wrote", out / f"{name}.jinja", len(text), "chars")


if __name__ == "__main__":
    main()
