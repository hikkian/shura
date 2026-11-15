# Lifelong memory / Долговременная память

## English

The candidate integration is opt-in. `memoryEnabled`, `memoryInject` and `lifelongCapture` are `false` by default. The candidate search settings are `memoryStem: true` and `memoryIndexBudgetTokens: 1500`; neither activates memory. `memoryPath: null` uses the standard local SQLite location.

Enable `memoryEnabled` in the gateway configuration to expose its memory API. The gateway owns writes and the generated index. In ShuraCode, enable `memoryEnabled` for the final `plugin/lifelong.ts` plugin and the TUI plugin, and set `SHURACODE_MEMORY_ENABLED=true` in the memory and rules MCP environments. Point `SHURA_LIFELONG_URL` at the same loopback gateway origin; if a custom `memoryPath` is used, give the rules MCP process that exact file through `SHURA_LIFELONG_DB`. Configure these in a candidate profile and start a new session.

The sidebar shows `memory: N records, rules: M, version V`. M counts enabled rules for the model, and V is the current personal-layer version. An unavailable gateway is displayed as unavailable. `/memory` and `/rules` show source, date and trust. Passing a gate never confirms a captured memory or enables a rule.

| Command | Effect |
|---|---|
| `/memory [query]` | List all records or search, with passports. |
| `/remember <text>` | Explicitly save a memory. |
| `/fix <key> -> <correction>` | Correct a record; retain the old version and passport. |
| `/forget <key>` | Soft-delete a record. |
| `/forget --purge <key>` | Export the deleted record before permanent deletion. |
| `/memory export [file]` | Export JSON Lines with passports and journal. |
| `/memory import <file>` | Import a named file; foreign material remains proposed quotes with an external source. |
| `/memory stats` | Show counts and disk/write statistics. |
| `/memory restore <key>` | Restore a soft-deleted record. |
| `/rules [key]` | Show personal rules and their passports; detailed history for a key. |
| `/rules versions` | Show versions of the personal layer. |
| `/rules rollback` | Show the person's terminal rollback command. |

Gateway reads: `GET /guardian/memory/list`, `search`, `read`, `export`, `stats`, `summary`. Writes use `POST` on `save`, `fix`, `forget`, `restore`, `import`, `resolve`. They require `memoryEnabled`. No memory call starts the model.

To capture conversation signals, additionally enable `lifelongCapture` in the gateway and in ShuraCode's final lifelong plugin. `/fix` and successful rule proposal/withdrawal operations send the person's words. For an ordinary correction, the model may call `capture_correction`, which takes no text argument and uses the actual latest user message. There is no background scan of conversations or files. The gateway's `POST /guardian/memory/capture` forces every captured record to `proposed` with zero confirmations; it cannot enable rules. The existing task/example capture plugin has its own opt-in setting.

`memoryInject` remains optional and off. When enabled together with memory, the gateway appends bounded, escaped data to user messages. Each message's decision (including no match) is frozen in gateway RAM and replayed in later histories. Previously injected text remains part of the historical prompt after a record is changed or forgotten; new messages search the updated store. ShuraCode freezes its memory index and personal-layer system text for the session. Changes become system instructions only in a new session. Prefix stability is checked with the fake llama-server; no real model or performance claim is involved. Restarting the gateway clears its RAM cache.

## Русский

Кандидатная интеграция включается явно. По умолчанию `memoryEnabled`, `memoryInject` и `lifelongCapture` равны `false`. Кандидатные настройки поиска — `memoryStem: true` и `memoryIndexBudgetTokens: 1500`; сами по себе они память не включают. `memoryPath: null` выбирает стандартный локальный SQLite-файл.

Включите `memoryEnabled` в конфигурации шлюза. В ShuraCode включите `memoryEnabled` у последнего плагина `plugin/lifelong.ts` и у TUI-плагина, задайте `SHURACODE_MEMORY_ENABLED=true` в окружении MCP памяти и правил. `SHURA_LIFELONG_URL` должен указывать на этот же loopback-шлюз; при нестандартном `memoryPath` передайте MCP правил этот файл через `SHURA_LIFELONG_DB`. Настраивайте кандидатный профиль и начинайте новую сессию.

Строка `memory: N records, rules: M, version V` показывает число видимых записей, включённых правил для модели и текущую версию личного слоя. Недоступность шлюза показывается явно. `/memory` и `/rules` сохраняют в выводе источник, дату и доверие. Пройденные ворота сами не подтверждают память и не включают правило.

| Команда | Действие |
|---|---|
| `/memory [запрос]` | Все записи либо поиск, с паспортами. |
| `/remember <текст>` | Явное сохранение записи. |
| `/fix <ключ> -> <исправление>` | Новая версия записи; прежняя остаётся в истории. |
| `/forget <ключ>` | Мягкое удаление. |
| `/forget --purge <ключ>` | Экспорт удаляемого перед окончательным удалением. |
| `/memory export [файл]` | JSON Lines с паспортами и журналом. |
| `/memory import <файл>` | Импорт названного файла; чужой материал — предложенная цитата с внешним источником. |
| `/memory stats` | Количество записей и статистика файла/записанных байтов. |
| `/memory restore <ключ>` | Возврат мягко удалённой записи. |
| `/rules [ключ]` | Правила с паспортами; для ключа — также история. |
| `/rules versions` | Версии личного слоя. |
| `/rules rollback` | Показ команды отката, которую выполняет человек. |

Чтение шлюза: `GET /guardian/memory/list`, `search`, `read`, `export`, `stats`, `summary`. Запись: `POST` на `save`, `fix`, `forget`, `restore`, `import`, `resolve`. Требуется `memoryEnabled`. Обращение к памяти модель не запускает.

Для захвата сигналов дополнительно включите `lifelongCapture` в шлюзе и последнем lifelong-плагине ShuraCode. `/fix` и успешные операции предложения/отклонения правила передают слова человека. При обычном исправлении модель может вызвать `capture_correction`: он не принимает текст от модели и берёт настоящее последнее сообщение человека. Фонового чтения диалогов или файлов нет. `POST /guardian/memory/capture` принудительно сохраняет `proposed`, с нулём подтверждений, и не включает правила. У прежнего плагина захвата задач/примеров собственная настройка включения.

`memoryInject` остаётся выключенной опцией. При включённой памяти шлюз дописывает ограниченный экранированный блок данных к сообщению пользователя. Решение для сообщения, включая отсутствие совпадений, фиксируется в ОЗУ и повторяется в следующих историях. Уже подставленный исторический текст остаётся прежним после исправления/удаления записи; новые сообщения ищут в обновлённой памяти. ShuraCode фиксирует системный индекс и личный слой на сессию. Новая сессия получает их обновления. Префикс проверяется фейковым llama-server без настоящей модели и обещаний производительности. Перезапуск шлюза очищает кеш в ОЗУ.
