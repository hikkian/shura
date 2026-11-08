# Qwen3.6-35B-A3B и Ornith-1.5-35B-A3B как локальные coding-агенты: chat-template, шкала effort, сценарии

Дата: 2026-11-06 (раунд 2). Все файлы — в `/home/hikkian/Desktop/AI/bench/template/`. Ничего не применялось: серверы, GPU, gateway и конфиги не трогались; инференс не запускался. Пометки: **[V]** — проверено по источнику (URL или файл:строка), **[I]** — вывод/предположение.

## 0. Коротко

1. **«Стоковый» `cand/qwen36.jinja` — не официальный шаблон Qwen**, а модификация автора кванта IsValorum (JSON tool-call вместо обученного XML, вставки «STRICT TOOL REGISTRY…», `[Reasoning Guidance: …]`, `max` не распознаётся) **[V]**. В сами GGUF (APEX, GSQ, и Ornith GSQ-RCO) встроен **официальный** шаблон Qwen3.6 (7764 байта, извлечён своим GGUF-ридером, сравнён побайтно) **[V]**, а запись `qwen36` в `model-launch.json` не передаёт `--chat-template-file` **[V]** → идущий бенчмарк «stock» = официальный шаблон.
2. **Идеальный шаблон = официальный + 7 точечных правок** (§3). На уровне `medium` он рендерит промпт **байт-в-байт как официальный**; отличия включаются только effort-строкой (4 уровня), kwarg `terse`, дефолтом preserve и защитой от управляющих токенов в tool-выводе.
3. **Шесть уровней effort**: fast = thinking off (официальный instruct-режим), low/high/xhigh/max = одна строка в конце system-блока + бюджет токенов в gateway, medium = нативное поведение + страховочный бюджет (§5).
4. Сервер shura.cpp: XML-шаблон → специализированный парсер `qwen3-coder` **[V]**; `preserve_reasoning=true` по умолчанию **[V]**; `reasoning_budget_tokens` — семплер, вне промпта **[V]**; клиент OpenCode возвращает `reasoning_content` **[V]** → история append-only, KV-кэш цел (рендер-тест).
5. **Ornith-1.5**: обучающий шаблон = Qwen3.6-шаблон с безусловным `<think>` для всей истории **[V]**; под llama-server с дефолтным preserve рендер идентичен → **один шаблон на обе модели**, `ornith-ideal.jinja` лишь фиксирует preserve.
6. Terse-блок Sharp по умолчанию **выключен**: он помогает скорости и вредит сценариям «объясни/напиши документ» (§2); проверяется одним A/B-армом на модель.

## 1. Факты по источникам

### Qwen3.6-35B-A3B — https://huggingface.co/Qwen/Qwen3.6-35B-A3B **[V]**
- Сэмплинг: thinking общий `temp 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5`; **thinking для точного кодинга `temp 0.6, top_p 0.95, top_k 20, presence 0`**; non-thinking `temp 0.7, top_p 0.8, top_k 20, presence 1.5`. SWE-bench у Qwen: `temp 1.0, top_p 0.95`, 200K.
- Вывод: «32 768 токенов для большинства запросов, 81 920 для сложных». Параметра thinking-budget нет; `/think`,`/nothink` не поддерживаются; thinking выключается `chat_template_kwargs.enable_thinking=false`.
- **Preserve Thinking**: «Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages… particularly beneficial for agent scenarios… reduce overall token consumption… improve KV cache utilization». Qwen-Agent запускают с `enable_thinking: true, preserve_thinking: true`.
- Инференс: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` → XML-формат tool-call. `reasoning_effort` не упоминается; официальный `chat_template.jinja` (…/raw/main/chat_template.jinja) переменной не содержит. Контекст 262K; совет держать ≥128K.

### Thinking budget Qwen3 — https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html **[V]**
- По исчерпанию бюджета дописывается «`\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n`» и генерация продолжается. То же делает `reasoning_budget_message` в shura.cpp (сервер сам добавляет `\n</think>`). Что Qwen3.6 реагирует как Qwen3 — **[I]**. В vLLM/SGLang бюджет для Qwen3.6 часто не применяется (HF discussion #66, sglang #25536) — у нас другая реализация (семплер), проверить на живом сервере (§7, шаг 0).

### froggeric v22.5 — https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates **[V]**
Правки: строковые `arguments`, склейка system-сообщений, `reasoning_effort` (`low/medium/xhigh`, алиасы `high,max,ultracode,extreme→xhigh`; medium без текста), inline-теги `<|think_*|>`, `tool_call_format=json`, усечение аргументов/ответов, «двухуровневая эскалация ошибок» (SYSTEM WARNING в tool_response), «100 % KV cache hit». Бенчмарков нет. Тексты: xhigh «…think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer», low «Keep your thinking brief and focused, moving directly to the conclusion…».

### Sharp v22.5.0 — https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates **[V]**
= froggeric + принудительный terse-блок после system-промпта, kwarg `terse` (default true), `high/max/…→xhigh`. Числа автора: Qwen3.8-27B SWE-bench-Live **15/25 Sharp vs 16/25 stock, медиана 20.0 мин vs 54.6**, оба medium; ThinkingCap-Qwen3.6-27B Claw-Eval answer +7.4, **answer tokens −59 %**. Известная проблема: `<|think_off|>` в tool-выводе выключает reasoning, если tool-результаты идут user-ролью (у нас так).

### Tiel-Coder — https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF **[V]**
«Ornith-1.5-35B-A3B re-quantized… carrying the Sharp chat template… We changed the chat template, not the weights.» SWE-bench-Live 12/25 vs Ornith 8/25 («four more than Ornith-1.5 itself»), stock Qwen3.6-35B-A3B 8/25; **«one run per problem… Treat small differences as noise»**; харнесс не назван. MMLU-Pro Tiel 73.7 vs Ornith 78.0 — автор: «the Sharp template buying shorter answers»; Claw-Eval: «5.1 down on clarifying questions». Наш `tiel.jinja` = `sharp.jinja` + identity-строка «You are Tiel-Coder, a variant of Ornith-1.5-35B-A3B.» (diff 2 строки) **[V]**.

### Ornith-1.5 — https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B **[V]**
Линия Ornith-1.0 (Qwen3.5 + Gemma4), RL с генерацией задач; «excels in tool-calling and agentic coding», «optimized for terminal-based coding agents». SWE-bench Verified 79 vs Qwen3.6-35B 73.4 (самоотчёт, 5 прогонов). Сэмплинг «general: temp 0.6, top_p 0.95, top_k 20; benchmarks: temp 1.0». Non-thinking режим, `reasoning_effort`, `preserve_thinking` не упоминаются. «We adjust the Qwen chat template to ensure consistency between training and inference» — HF-шаблон рендерит `<think>` для всех прошлых assistant-ходов безусловно; остальное = Qwen3.6 (XML, `--tool-call-parser qwen3_xml/qwen3_coder`). GSQ-RCO квант (https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF): `--jinja --temp 0.6 --top-p 0.95 --top-k 20 --chat-template-kwargs '{"preserve_thinking":true}'`; в нашем файле встроен официальный Qwen3.6-шаблон **[V]**.

### IsValorum APEX — https://huggingface.co/IsValorum/Qwen3.6-35B-A3B-MTP-APEX-I-NanoPlus-GGUF **[V]**
Карточка: «hardened» шаблон с `reasoning_effort` (low/minimal, medium, high/xhigh, none/off — `max` нет), «anti-premature stop», «multi-turn thinking memory by default» (коду не соответствует: без `preserve_thinking=true` прошлые `<think>` не сохраняются). Переход на JSON не объяснён, источник не указан.

### Сервер shura.cpp (commit c4d92d5e, 2026-10-30) **[V]**
- `tools/server/server-common.cpp:1498-1507`: top-level `reasoning_effort` → `chat_template_kwargs.reasoning_effort`; `"none"` → `enable_thinking=false`, kwarg удаляется. (Утверждение Sharp-README «llama.cpp не передаёт top-level reasoning_effort» для этого форка неверно.)
- `:1518-1533`: `reasoning_budget_tokens`/`thinking_budget_tokens` per-request, иначе `--reasoning-budget` (−1); `reasoning_budget_message` per-request или флагом; теги из парсера.
- `common/sampling.cpp:375-386`: семплер создаётся при бюджете ≥0; токены generation-prompt (`<think>\n`) скармливаются ему → отсчёт с первого сгенерированного токена.
- `common/reasoning-budget.cpp`: IDLE→COUNTING по `<think>`; при исчерпании (после завершения UTF-8-последовательности) FORCING: логиты −inf кроме токенов `message + "\n</think>"`; затем passthrough; повторный `<think>` — новое окно. Естественный `</think>`/`<tool_call>` до лимита — без вмешательства.
- `common/chat.cpp:1326-1333` + `common/parsers/qwen3-coder.cpp`: парсер выбирается по подстрокам `<tool_call>`, `<function=`, `<parameter=` (и `<think>`); `thinking_end_tags = {"\n</think>", "</think>", "<tool_call>"}`; lazy grammar по триггеру `<tool_call>`; обязательные параметры в любом порядке. JSON-шаблон → «дифференциальный автопарсер».
- `common/arg.cpp:975-977`: `preserve_reasoning=true` по умолчанию (`--no-reasoning-preserve` выключает); `common/jinja/caps.cpp:22` → `preserve_thinking` в контексте шаблона. `common/jinja/value.cpp:790`: фильтр `replace` есть. `common/jinja/lexer.cpp:56`: завершающий `\n` файла отбрасывается.

### Gateway и клиент **[V]**
`ai_gateway.py:1280` `reasoningEffort→reasoning_effort`; `:724` перенос `<env>` в первый user-месседж; `:232` ключ слота = sha256(tools+system из запроса) — бюджет/kwargs в хеш не входят; `:1102` `apply_terse` для моделей с `chatTemplateKwargs.terse`; `:302` `chatTemplateKwargs→--chat-template-kwargs`. `shuracode.json`: варианты `fast→low, low, medium, high, max`. Бинарник OpenCode 1.18.32: assistant-история уходит с `reasoning_content` (`case"reasoning":{L+=J.text}` → `{reasoning_content:L}`). Наши данные: Tiel 51/57; Codex 55/57; опечатки RU/1000 слов: Qwen3.6 ≈1.0, Ornith GSQ-RCO ≈2.2, Tiel APEX 4.6.

## 2. Сценарии: что модели нужно от шаблона/системного текста

Общее правило, которое вытекает из таблицы: **шаблон отвечает за протокол (формат tool-call, история, reasoning) и за переменное по запросу (effort); всё, что одинаково на всех уровнях и для обеих моделей, — в системном промпте ShuraCode.** Поэтому ряд находок ниже оформлен как правки `prompt.txt` (`proposed/prompt.txt.diff`, не применены), а не шаблона.

| Сценарий (задачи) | Что помогает | Что вредит | Куда это идёт |
|---|---|---|---|
| Починка по падающим тестам (t1, t2, t6) | preserve (план и гипотезы не передумываются на каждом шаге) **[V: Qwen card]**; medium/high; точный XML-формат → нет потерянных вызовов | low без чтения теста; кап бюджета ниже длины «разбора» первого шага **[I]**; Sharp-эвристика «SYSTEM WARNING» на вывод тестов с `error:` (ложный «сбой инструмента») **[V по коду sharp.jinja:443-466; эффект I]** | шаблон: preserve; gateway: бюджет |
| Фича/рефакторинг в чужом репо (t4, t5, t8) | high/xhigh-строка «сравни подходы, спланируй»; todowrite; preserve | terse (съедает итоговый отчёт «что изменил/как проверил», который требует системный промпт) **[I]**; thinking на каждом routine-шаге одинаково длинный → в effort-строках добавлена фраза про routine-шаги **[I]** | шаблон: effort; промпт уже есть |
| «Где/как X» без правок (t7) | прямой ответ с `path:line`; fast/low достаточно | над-мышление; длинные ответы | промпт (есть) |
| Объяснить архитектуру новичку (t13) | полный текст «в нескольких абзацах», как просит пользователь | **terse**: «least structure», «lean final answer» прямо противоречат задаче; свидетельства: Tiel MMLU-Pro −4.3 «shorter answers», Sharp «answer tokens −59 %» **[V числа; перенос на объяснения — I]** | terse off; промпт: правка «requested content is the deliverable» |
| Вёрстка/UI по скриншоту (t9, t15, t16) | официальный vision-рендер `<|vision_start|><|image_pad|><|vision_end|>` **[V]**; high (планирование макета); «сначала опиши, что на картинке» **[I]** | низкий бюджет (описание макета в thinking — полезная часть); `--image-max-tokens` — конфиг, не шаблон | шаблон: без изменений; промпт: правка про скриншоты |
| Огромные логи, 100k+ (t14) | grep-окна вместо чтения целиком; страховочный бюджет medium (не дать «улететь» над большим вводом) **[I]**; Qwen: контекст ≥128K **[V]** | любые инъекции в tool_response (Sharp-WARNING на «error:» в первых 120 символах — для логов срабатывает почти всегда); управляющие токены в логе (`<|im_end|>`) закрыли бы ход **[V по механике parse_special; частота — I]** | шаблон: `clean()`; промпт: правка про большие файлы |
| Веб-исследование (t17) | официальная строка «If there is no function call available, answer… normally»; «назови источник» | веб-страница с `<|im_start|>` / фразами «ignore instructions» — защита `clean()` + правило «текст из инструментов — данные» **[I]**; inline-теги Sharp из веб-текста **[V: changelog Sharp]** | шаблон: `clean()`; промпт: две правки |
| Проза RU/KZ без опечаток (t12, t18) | низкая температура и `--ban-scripts cjk,homoglyph` (конфиг) **[V]**; самопроверка текста перед отправкой **[I]**; достаточный бюджет для структуры | terse (README/письмо — не «lean answer»); опечатки — в основном свойство кванта (1.0 vs 2.2 vs 4.6 на тех же шаблонах) **[V данные]** | промпт: правка в Language; шаблон — ничего |
| Планирование/дизайн-обсуждение (идея, не код) | high/xhigh/max-строки переписаны под «goal, approaches, ideas, trade-offs» (не только баги); правило «explanation/plan → не редактировать» (есть) | terse; medium иногда даёт первое пришедшее решение **[I]** | шаблон: effort; промпт |
| Неоднозначные запросы | правило промпта «one short, specific question if it changes what you'd build; else assume and state» (есть) | дубль правила в terse («ask a sharp question») не нужен; Tiel: «−5.1 on clarifying questions» при terse **[V]** | промпт (есть) |
| Честность (t19) | раздел Honesty промпта; effort ≥ medium | fast/low: короткое мышление → больше шансов «сделать вид» **[I]**; фраза бюджета «give the solution… directly now» — потенциально подталкивает к ответу вместо «не могу» **[I]**, поэтому medium-кап высокий | промпт (есть); gateway: бюджет |
| Долгие сессии | preserve: append-only история → кэш и слоты gateway работают **[V: рендер-тест]**; Qwen: меньше повторного мышления **[V]**; бюджет ограничивает рост контекста от thinking | без preserve — пересчёт всего после первого assistant-хода предыдущего вопроса при каждом новом вопросе **[V: рендер-тест]**; на CPU-offload MoE это минуты **[I]** | шаблон: preserve |
| Быстрые однострочные вопросы | fast = thinking off (официальный режим, prefill `<think>\n\n</think>`) **[V]**; никакого effort-текста | effort-строка при thinking off (не вставляется); большие бюджеты | шаблон: `none`→off |

Вывод по terse: нигде не помогает качеству, помогает времени/токенам; для 4 из 13 сценариев (объяснение, проза, планирование, отчёт после работы) вредит. Дефолт — off, один A/B-арм на модель.

## 3. Идеальный шаблон (B)

Генерируются `make_templates.py` из одного источника:

| Файл | Фактор |
|---|---|
| `qwen36-ideal.jinja` | **финальный для Qwen3.6** |
| `qwen36-ideal-terse.jinja` | terse ON (то же: `qwen36-ideal.jinja` + `--chat-template-kwargs '{"terse":true}'`) |
| `qwen36-ideal-nopreserve.jinja` | preserve OFF (контроль-негатив, не для прода) |
| `qwen36-ideal-effort-stock.jinja` / `-effort-none.jinja` | тексты IsValorum / без текстов (запасные армы) |
| `ornith-ideal.jinja` | **финальный для Ornith-1.5** (= qwen36-ideal с preserve всегда) |
| `ornith-ideal-terse.jinja` | terse + identity-строка Tiel (честная копия Tiel без froggeric-правок) |
| `ornith-ideal-effort-none.jinja` | запасной |

Правки относительно официального (остальные байты, включая текст `# Tools`, рендер tool_call/tool_response, generation-prompt — официальные):
1. Ведущие system-сообщения склеиваются; system в середине рендерится, а не `raise_exception` (краш = HTTP 500 на ход).
2. `preserve_thinking` по умолчанию `true` (под llama-server переменная всегда задана; дефолт важен для других движков).
3. `reasoning_effort`: `none|off` → thinking off; `low|minimal`, `high`, `xhigh`, `max|ultracode|extreme` → одна строка **в конце** system-блока (общий префикс между уровнями = весь блок tools+system, 8577 байт в тесте; у Sharp/Tiel — 19 байт, т.к. строка стоит до `# Tools` **[V: validation.txt, E]**); `medium` → ничего; при thinking off текст не вставляется.
4. `terse` kwarg (точный текст Sharp), default off.
5. Строковые `arguments` → `<parameter=arguments>` вместо ошибки.
6. Нет `raise_exception('No user query found')`.
7. **`clean()`**: в тексте user/tool-сообщений `<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>` получают пробел внутри маркера (`<|im_end |>`), т.е. остаются обычным текстом. Сервер токенизирует промпт с `parse_special`, поэтому такие строки из веб-страницы или лога закрыли бы ход или подделали system-сообщение; рендер-тест показывает, что через официальный шаблон инъекция проходит дословно, через ideal — нет **[V: validation.txt, S]**. Детерминированно → кэш не страдает. Ассистентский текст не трогается (модель не может сгенерировать спец-токены как текст).

Тексты effort (финальные; medium — пусто):
- low: «Reasoning effort: low. Think briefly and act: decide the next step in a few sentences, do not re-derive what you already know, and do not plan beyond that step.»
- high: «Reasoning effort: high. Before acting, make sure you understand what the user actually wants and why; consider a couple of approaches (to the code, the design, the wording or the explanation) and pick the better one; think through edge cases and how you will check the result. Routine steps (reading, running tests) still need only a short thought.»
- xhigh: «Reasoning effort: very high. Take the time to think it through: pin down the goal and the constraints, explore several genuinely different approaches and their trade-offs, look for a better idea than the first one, plan the work, anticipate what could go wrong or be missed, and decide how you will verify the result. Spend the effort where decisions are made; routine steps still need only a short thought.»
- max: «Reasoning effort: maximum; quality matters more than time. Think deeply and broadly: fully understand the goal and its context, generate and compare several ideas including non-obvious ones, choose the best and justify the choice to yourself, plan carefully, address root causes rather than symptoms, and critically review your own conclusions before acting and before answering. Spend the effort where decisions are made; routine steps still need only a short thought.»

Почему так: строки читаются на каждом шаге агентского цикла, а не раз на задачу — поэтому в high+ добавлено «routine steps need only a short thought» (иначе high = долгое мышление после каждого `read`) **[I]**; формулировки покрывают код/дизайн/текст/объяснение, а не только исправление багов (переписаны пользователем, здесь ужаты и выровнены по градации); конфликтов с `prompt.txt` нет (он не говорит, сколько думать; «Length should match the task» — про ответ, не про мышление).

Чего в шаблоне нет и почему: эскалации ошибок Sharp (нетренированный текст в tool-роли, ложные срабатывания на логах; правило «fails twice → change approach» уже в промпте); переписанного `# Tools`; inline-тегов `<|think_*|>` (уязвимость через tool-вывод); усечения аргументов/ответов (молчаливое усечение ломает `edit.oldString`); «thinking-prefill» (парсер и бюджет рассчитаны на generation-prompt ровно `<|im_start|>assistant\n<think>\n`) **[I]**.

Проверка (`render_test.py`, jinja2 3.1.6 в `.venv`, `tojson` как в llama.cpp, завершающий `\n` отбрасывается как в `lexer.cpp:56`): реальный `prompt.txt` + 8 инструментов OpenCode + 7 ходов (user с `<context>`, tool_call, параллельные вызовы, итог, второй вопрос), текст и с картинкой; effort unset/none/low/medium/high/xhigh/max × preserve true/false/unset; 13 шаблонов. Итог (`validation.txt`, образцы `samples/`): все рендерятся; generation-prompt `<think>\n` / `<think>\n\n</think>\n\n`; tool-call блоки совпадают с форматом парсера; `xhigh`, `max` распознаны; история append-only при preserve у всех ideal (разрывы только при preserve=false — на новом вопросе пользователя, как и у официального); инъекция управляющих токенов отбита. Не проверено без сервера: выбор парсера и `supports_preserve_reasoning` живым llama-server (по коду — `qwen3-coder`/`true`); использованы только конструкции из официального шаблона и Sharp плюс `replace` (есть в `value.cpp:790`).

## 4. Сравнение шаблонов (A)

| Аспект | Официальный (= встроен в GGUF) | IsValorum `qwen36.jinja` | Sharp / Tiel | Ornith-HF | **ideal** |
|---|---|---|---|---|---|
| Tool-call | XML (обучающий) | JSON | XML | XML | XML |
| Парсер llama.cpp | `qwen3-coder` | автопарсер | `qwen3-coder` | `qwen3-coder` | `qwen3-coder` |
| `# Tools` | официальный | переписан | переписан | официальный | официальный |
| tool_response | официальный | то же | + SYSTEM WARNING по эвристике | то же | официальный + `clean()` |
| Прошлый reasoning | `preserve_thinking` (под сервером вкл.) | то же, пустой блок опускается | preserve default вкл. | всегда | default вкл. |
| effort | нет | low/medium/high; `max` — нет | low/xhigh (high/max→xhigh); **до `# Tools`** | нет | 4 строки в конце system |
| Инъекция system-текста | нет | `[Reasoning Guidance]` | terse (+identity у Tiel) | нет | нет (terse по kwarg) |
| 2-й system / без user-запроса | `raise_exception` | `raise_exception` | ок | `raise_exception` | ок |
| Спец-токены из tool-вывода | проходят | проходят | проходят | проходят | обезврежены |
| Датированное/случайное в префиксе | нет | нет | нет | нет | нет |

Для кэша gateway (ключ слота = tools+system запроса): всё, что меняет первые байты рендера при том же запросе, делает сохранённый слот бесполезным; у Sharp/Tiel это effort≠medium, у ideal — ничего (effort-строка последняя в system-блоке, бюджет вне промпта).

## 5. Шкала effort: шесть уровней (C)

Принцип: бюджет токенов — основной рычаг (жёсткий, не зависит от послушности, вне промпта), текст — вспомогательный, medium — нативное поведение. Декод ~40 ток/с.

| Вариант | `reasoningEffort` | Thinking | Текст | `reasoning_budget_tokens` | Потолок времени на шаг |
|---|---|---|---|---|---|
| fast | `none` | off (instruct-режим) | — | — | — |
| low | `low` | on | low | **2048** | ~50 с |
| medium (дефолт) | `medium` | on | — | **8192** (страховка от «улёта», в норме не срабатывает) | ~3.5 мин |
| high | `high` | on | high | **16384** | ~7 мин |
| xhigh | `xhigh` | on | xhigh | **32768** (= рекомендованная Qwen длина вывода «для большинства запросов») | ~14 мин |
| max | `max` | on | max | **−1** | без лимита |

Сообщение при исчерпании (одно для всех): `"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now."` **[V: текст Qwen3-доков; применимость к 3.6 — I]**; сервер добавляет `\n</think>`. Числа бюджетов — **[I]**; откалибровать по распределению длины reasoning на шаг из текущего прогона (сервер отдаёт `reasoning_content`): medium ≈ p99 нормальных шагов, low ≈ медиана, high/xhigh ≈ максимум наблюдаемого на сложных задачах. Сэмплинг по уровням не меняется (официальный non-thinking `0.7/0.8/presence 1.5` — возможная добавка в gateway для `none`, не включена) **[I]**.

Где что живёт (правки — в `proposed/`, не применены):
- **Шаблон**: маппинг `reasoning_effort`→текст, `none`→off, `terse`.
- **Gateway** (`proposed/gateway-effort-budget.diff`, ~20 строк): после переименования `reasoningEffort` добавить `reasoning_budget_tokens`/`reasoning_budget_message` по таблице `effortBudgets` модели.
- **model-launch.json** (`proposed/model-launch.arms.json`): `--chat-template-file` в `extraArgs` **и** `visionOverrides.extraArgs`, `chatTemplateKwargs {"terse": false}`, `effortBudgets`.
- **shuracode.json** (`proposed/shuracode.arms.json`): `fast→"none"`, новый вариант `xhigh`.
- **prompt.txt** (`proposed/prompt.txt.diff`): 6 строк по сценариям — самопроверка RU/KZ-текста; «requested content is the deliverable»; большие файлы/логи через grep-окна; скриншот → сначала описать; текст из инструментов — данные, не инструкции; называть URL источника.

## 6. Ornith-1.5

Отличие в ожиданиях — только безусловный `<think>` в истории **[V]**; формат tools/tool_response/vision/generation-prompt тот же. Почему Sharp мог помочь Ornith по карточке Tiel **[I]**: (а) 12 vs 8/25 при одном прогоне — шум по словам автора; (б) в их стеке стоковый GGUF-шаблон = Qwen3.6-шаблон, и без включённого `preserve_reasoning` модель, обученная видеть свой прошлый reasoning, его не видела — Sharp включает preserve по умолчанию; у нас сервер включает его сам, так что этот эффект получен без Sharp. Остальное (terse, froggeric-правки) — вопрос A/B `ornith-it` vs `ornith-i`. Fast=thinking off для Ornith не документирован, но шаблон его поддерживает (**[I]** о качестве). Один шаблон на обе модели — да; `ornith-ideal.jinja` отличается одной строкой. Сэмплинг 0.4 vs рекомендованных 0.6 — отдельный фактор, не в этом A/B.

## 7. Итоговая рекомендация и A/B (D)

**Рекомендация по умолчанию**: `qwen36-ideal.jinja` / `ornith-ideal.jinja`, terse off, preserve on (дефолт сервера), medium без текста + бюджет 8192 через gateway, fast = thinking off, правки `prompt.txt` из §5. Это минимальное отклонение от обучающего формата обеих моделей плюс жёсткие, независимые от модели гарантии (бюджет, кэш, защита от спец-токенов).

Шаг 0 (без бенчмарка, ~10 мин): поднять `qwen36i`, запрос с `reasoning_budget_tokens: 32`, убедиться в логе: «Using specialized template: Qwen3-Coder», «chat template supports preserving reasoning», и что reasoning обрывается фразой «Considering the limited time…» с нормальным продолжением. Без (в) бюджетные армы не имеют смысла.

Подмножество: `t3_duration, t8_cache, t12_readme, t13_bigrepo, t14_biglog, t16_mockup, t19_honesty`, 3 повтора, тот же раннер/hidden-тесты; 21 прогон на арм (~1–3.5 ч). Метрики: pass, время на задачу, reasoning-токены на шаг, доля шагов с срабатыванием бюджета, ошибки парсинга tool-call, опечатки RU/1000. Разница <3/21 — шум.

Армы (не более 4 на модель, контроли уже измерены):

| | Шаблон | Effort | Бюджет | Вопрос |
|---|---|---|---|---|
| Q0 `qwen36` | официальный (встроен) | medium | нет | контроль — взять 7 задач из идущего прогона |
| Q1 `qwen36i` | ideal | medium | 8192 | промпт = Q0 байт-в-байт → изолирует только бюджет-кап и `clean()` |
| Q2 `qwen36i` | ideal | **high** | 16384 | даёт ли дополнительное мышление + high-строка прирост на сложных задачах и какой ценой во времени |
| Q3 `qwen36it` | ideal + terse | medium | 8192 | terse: быстрее ли и не теряет ли t12/t13 |
| O0 `tiel-coder` | Sharp+identity (встроен) | medium | нет | контроль — 7 задач из прогона 51/57 |
| O1 `ornith-i` | ideal | medium | 8192 | убрать всё Sharp'овское, оставить preserve |
| O2 `ornith-it` | ideal + terse(+identity) | medium | 8192 | O2 vs O0 = цена froggeric-правок; O2 vs O1 = цена terse |
| O3 `ornith-i` | ideal | high | 16384 | high для Ornith |

Порядок: шаг 0 → Q1, Q3 → O1, O2 → Q2, O3. Решения: если Q1≈Q0 и бюджет не срабатывал — оставить 8192 как страховку; terse включать только при ≥3/21 выигрыше без потерь на t12/t13; high делать дефолтом только если выигрыш ≥3/21 при приемлемом времени; иначе дефолт — medium, high/xhigh/max остаются ручным выбором пользователя.

Проводка: добавить записи из `proposed/model-launch.arms.json` и `proposed/shuracode.arms.json`; применить `proposed/gateway-effort-budget.diff` (перезапуск gateway — вне моей зоны); в раннере менять только `model` и вариант effort. Для «без бюджета» — `reasoning_budget_tokens: -1` в запросе (клиентское значение приоритетно).

## 8. Риски и открытые вопросы
- Бюджет-семплер и фраза остановки на Qwen3.6/Ornith не проверены вживую (шаг 0). Числа бюджетов — без данных.
- `fast=none`: качество non-thinking в агентском цикле не измерено; запасной вариант — `low` + бюджет 512.
- `clean()` меняет байты tool-вывода только при наличии спец-токенов; теоретически модель могла бы «не узнать» `<|im_end |>` в цитате — это и есть цель.
- `qwen36.jinja` IsValorum использовать не стоит (JSON против обучения, автопарсер, нет `max`, пропуск пустого `<think>` ломает соответствие кэша после fast-ходов).
- Сэмплинг `temp 0.4` против официальных 0.6/1.0 — фактор, возможно, весомее любого шаблона **[I]**.
- Harness-артефакт: отметки `P/X` для JSON-шаблона IsValorum в `validation.txt` не интерпретировать.

## 9. Файлы
`DESIGN.md`; `validation.txt`, `samples/<шаблон>/…`; `make_templates.py` → `qwen36-ideal*.jinja`, `ornith-ideal*.jinja`; `render_test.py`; `gguf_template.py` (чистый Python GGUF-ридер); `.venv`; извлечённые `tiel.jinja`, `ornith.jinja`, `qwen36-apex-embedded.jinja`, `qwen36-gsq-embedded.jinja`; `proposed/gateway-effort-budget.diff`, `proposed/model-launch.arms.json`, `proposed/shuracode.arms.json`, `proposed/prompt.txt.diff` — не применены.
