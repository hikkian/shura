# NOTE0 — отчёт

Сделана начальная заготовка инструмента NOTE0 в разрешённых путях.
В истории до этой карточки NOTE0-кода не было; fork.py и proxy.py уже содержали необходимую основу.
Добавлены определения шести типов правил и функции сборки запросов/классификации.
Добавлены начальные CLI заготовки запуска, анализа и счётчика, а также синтетические тесты.
Тесты `python3 -m unittest tests.test_note -v`: 5 тестов прошли.
Работа не завершена: реальный набор пар и строгое правило анализа требуют дописывания.
Точный body уже можно записывать proxy.py с `AGENTIC_EVAL_LOG_REQUESTS=1`; GPU здесь не запускался.
Статус: частично выполнено; ниже перечислены обязательные продолжения NOTE0/NOTE1.

## Что было и что изменено

До задачи уже существовали `fork.py` с handover на шаге k, `FORK_DEBUG` для диагностики и `proxy.py` с полным логированием request body при `AGENTIC_EVAL_LOG_REQUESTS=1`. Нового hook не добавлял. В git-истории NOTE0-коммитов и реализации NOTE0 не обнаружено. Созданы `note.py`, три простых CLI точки входа и `tests/test_note.py`.

## Что сделано

Есть шесть структурированных заготовок типов уроков, regex-классификатор, проверка буквальных параметров, pure request builder для режимов, bootstrap-оценка и начальная агрегация. В CLI предусмотрен запрет портов 8080 и 18090. Сокеты не использовались.

## Пары и важные ограничения

Кандидатная генерация в текущей заготовке не готова для эксперимента: точные bodies отсутствуют, пары не строятся по фактической границе request, taught/repeat стороны не разведены по идеям, контроли не созданы; поэтому числа нельзя считать валидными и сухой прогон не выполнен. Счётчик реальных записей пока заглушка, он не читает хранилища. Правило анализа реализовано лишь частично: нужно добавить контроль false firing, валидность усечений, доверительные границы и расчёт G_search/стоимости.

## Как получить точное тело запроса

`proxy.py` при `AGENTIC_EVAL_LOG_REQUESTS=1` пишет `<log>.bodies.jsonl`, включая system prompt, tools, messages. Для шага k дешевле восстановить trajectory через FORK handover на fake llama, чем повторять генерацию до k на GPU. В NOTE1 следует поднять fake llama из `shura-qa/fake_llama/fake_llama_server.py`, запустить `fork.py run --log <run.agent.log> --task <task-id> --k <k> --upstream http://127.0.0.1:<fake-port> --out <dir>` с `FORK_DEBUG=<debug-log>` и `AGENTIC_EVAL_LOG_REQUESTS=1`, направив proxy к fake llama; извлечь body на первом forwarded запросе после k и проверить его инструменты/историю. Этот сценарий здесь не запускал. Перед NOTE1 надо сверить фактические аргументы fake llama и harness в shura-qa, выбрать только `part=search` по manifest и сохранить body вместе с run/step ID. Если fake-сервер не принимает handover, NOTE1 придётся повторно выполнять задачи на GPU, что потребует часов генерации.

## Проверка и незавершённое

`python3 -m unittest tests.test_note -v` — 5 тестов прошли (builders, regex, параметр урока, синтетический analysis). Runner против fake llama не запускался: по среде сокеты запрещены. Dataset, body capture, настоящий счётчик, полноценный analysis, dry-run с достоверными числами и проверка NOTE1 GPU-метода не сделаны. Поэтому критерии приёмки не выполнены и итог частичный.

## Файлы для коммита

- `bench/agentic-eval/note/note.py`
- `bench/agentic-eval/note/note_run.py`
- `bench/agentic-eval/note/count_real_pairs.py`
- `bench/agentic-eval/note/note_analyze.py`
- `bench/agentic-eval/note/REPORT-NOTE0.md`
- `tests/test_note.py`
