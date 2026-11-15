# EXAM200a — отчёт

Дата: 2026-11-13. Итог: **partial**.

## Что уже было в репозитории

До этой работы существовали `exam/build_v3r1.py`, overlay `exam/prompt-additions.json` (109 уточнений), манифест v3r1 и overlay-сборщик `exam/build_extra_r1.py` с 21 дополнительной фразой. Сборка overlay уже выдавала 465 задач с 142 правками входов, 104 правками раскладки и 30 правками документации. Наборы r1 уже имелись на диске.

В рабочем дереве до моих изменений уже были незакоммиченные изменения в `build_tasks_v2.py` и `tests_v2/test_build_tasks_v2.py`: в частности, JS grader стал копировать проект через `fs.cpSync`, обходить файлы тестов, а тест проверял эту конструкцию. Эти изменения сохранены. Другие незакоммиченные файлы карточек и `reports/ll6a/` также не трогались.

## Что изменено

В имеющемся генераторе v2 тест документации теперь проверяет, что заглушка `to be completed` удалена, а не требует точную фразу `Validation is explicit`. Условие многофайловой задачи теперь говорит обновить `README.md` и заменить заглушку в `docs/design.md`. Условие рефакторинга теперь называет фасад, core-файл и старый файл для удаления на Python и JS. Добавлен тест этих требований.

## Хеши и сборка v3r1

Команды из `bench/agentic-eval`:

```text
python3 exam/build_v3r1.py --out exam/tasks_v3r1
python3 exam/build_extra_r1.py
```

Сборка выдала 465 задач; правки: inputs=142, docs=30, layout=104, без правок=243. Для проверки хешей использовалось сравнение всех записей `exam/build-v3r1.log.json` с `exam/manifest-v3r1.json` по `id`, `hidden_sha256` и `prompt_sha256`:

```text
manifest rows 465, build rows 465
hidden mismatch 0
prompt mismatch 0
missing 0
```

Воспроизводимая команда для этой сверки — запустить `exam/build_v3r1.py --out exam/tasks_v3r1`, затем сравнить JSON-поля скриптом:

```python
import json
from pathlib import Path
m = json.loads(Path('exam/manifest-v3r1.json').read_text())['tasks']
b = json.loads(Path('exam/build-v3r1.log.json').read_text())
by = {x['id']: x for x in b}
assert len(m) == len(b) == 465
assert all(x['id'] in by and x['hidden_sha256'] == by[x['id']]['hidden_sha256']
           and x['prompt_sha256'] == by[x['id']]['prompt_sha256'] for x in m)
```

## Проверки

Успешный запуск:

```text
python3 -m unittest tests_v2.test_build_tasks_v2 -v
Ran 6 tests in 0.208s
OK
```

`git diff --check` завершился без замечаний.

Проверка всех задач была запущена так:

```text
nice -n 19 python3 exam/build_v3r1.py --out /dev/shm/exam200a-tasks-v3r1 --validate
```

Сборка прошла, но проверка сообщила 465 проблем: у всех задач `solution_rc=1` и `start_rc=1`. Диагностический запуск одной задачи показал причину до старта тестов:

```text
bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted
```

Поэтому проверки эталонов и стартовых файлов в этом sandbox не состоялись; эти exit code не трактую как падения самих задач. Временные файлы проверки удалены.

## Не сделано и почему

1. **Перенос всех 109 уточнений в генераторы/семейства.** В репозитории есть материализованный набор `tasks_v3` и builder overlay, но не найден исходный генератор этих 120 семейств. Доступные `families_v2/catalog.py` и `families_v2/` описывают отдельный набор из 20 семейств; ни одно имя семейства не пересекается с 109 записями в `prompt-additions.json`. Применить эти уточнения к другому каталогу было бы неверно. Соответствие Python/JS условий также не сводилось.
2. **Мутанты tests-only.** 111 grader-файлов обнаружено в v3; не запускались контрактные тесты sibling `implement` против мутантов. Таблица с числом неконтрактных и добавленных в v3r1 требований отсутствует — это обязательная незавершённая часть.
3. **Генератор tasks_lh.** `build_tasks_lh.py` только материализует `start/` из локальных git-архивов, а не строит `task.json`. Фразы для LH уже применяет `build_extra_r1.py`. Изменять task-папки запрещено, а безопасный путь интеграции в генератор метаданных не найден. Сверка prompt-ов перестроенного LH набора с `exam/tasks_lh_r1` отдельно не проводилась.
4. **Полная проверка эталонов/стартовых файлов** заблокирована запретом `NETLINK_ROUTE` внутри `bwrap`.

Сборки r1 в `exam/` созданы по инструкции; исходные `tasks_v3/`, `tasks_control_public/` и папки задач `tasks_lh/` не изменялись.

## Файлы для коммита

Изменены мной:

- `bench/agentic-eval/build_tasks_v2.py`
- `bench/agentic-eval/tests_v2/test_build_tasks_v2.py`
- `bench/agentic-eval/exam/REPORT-EXAM200a.md`

Ранее существовавшие изменения в этих двух файлах сохранены и входят в текущее незакоммиченное состояние. Новые r1-наборы сборки не перечислены для коммита.
