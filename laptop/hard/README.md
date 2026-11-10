# tasks_hard: 6 тяжёлых задач на рассуждение

Задачи для сравнения уровней reasoning effort. Формат как у `tasks_heldout/` (см. `bench/template/gen8/TASK-FORMAT.md`):
в каждой папке `prompt.txt`, `repo/`, `hidden_test.py`, `reference/`. Скрытые тесты написаны до любых запусков моделей и не меняются
(`chmod a-w`, хеши в `HARD_HASHES.txt`).

| задача | язык | тема |
|---|---|---|
| hd1_shift_hours | Python | расписание работы и время |
| hd2_js_cache | Node (ESM, без пакетов) | кэш с лимитом и сроком жизни |
| hd3_calc_parser | Python | разбор и вычисление выражений |
| hd4_recurrence | Python | правила повторения событий |
| hd5_patch_apply | Python | применение unified diff |
| hd6_js_allocate | Node (ESM, без пакетов) | деление денег по долям |

Только стандартная библиотека. Тесты лёгкие (секунды, без больших данных).

## Что лежит в reference/

`reference/` бенчем не используется. В нём три подкаталога-оверлея (только изменённые файлы поверх `repo/`):
`solution/` (проходит всё), `medium/` (проходит часть), `naive/` (небрежное решение). Они доказывают, что скрытые тесты различают уровни.

## Проверка эталонов

    python3 verify_reference.py            # все задачи
    python3 verify_reference.py hd3_calc_parser

Скрипт копирует `repo/` + оверлей + `hidden_test.py` во временный каталог (`$TMPDIR`), запускает `python -m unittest -v hidden_test`
и видимые тесты, печатает долю пройденных проверок. Каждая проверка — отдельный `test_*` метод, поэтому доля считается.

## Запуск скрытого теста вручную

    cp -r hdN_.../repo /tmp/x && cp hdN_.../hidden_test.py /tmp/x && cd /tmp/x && python3 -m unittest -v hidden_test

(JS-задачи: `hidden_test.py` сам запускает `node`, пишет временный `_hidden.mjs` в текущий каталог.)

## Как подключить к бенчу

Сейчас `bench.sh` ищет задачу только в `tasks/` и `tasks_heldout/`, а `score2.py` — в `tasks/` и `tasks_heldout/`; `tasks_hard/` они не видят.
Чтобы запускать эти задачи, нужно либо добавить `tasks_hard` в обе строки поиска каталога (`TD=...` в `bench.sh`, `TD=next(...)` в `score2.py`),
либо положить символические ссылки на `hdN_*` в `tasks_heldout/`. Ни то, ни другое здесь не сделано.
`score2.py` считает прогон успешным только при полном прохождении `hidden_test` (и видимых тестов); долю проверок он не записывает —
в `hidden_err` попадают последние 300 символов вывода unittest.
Таймаут `score2.py` на скрытый тест — 120 с, тесты укладываются в несколько секунд.
