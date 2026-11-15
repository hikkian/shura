# EXAM200b: источники микрозадач (только открытые)

Проверено 2026-11-13 командами `gh repo view` и `git ls-remote origin`.

| Репозиторий | Адрес | Видимость | Лицензия | Ветка (проверенная) | Коммит на публичной ветке |
|---|---|---|---|---|---|
| barid | https://github.com/hikkian/barid | PUBLIC | MIT | `origin/main` | `bf3da353577537294fa43e525a2a3d3bdb1246d8` (совпадает с локальным `main`, 0 непубличных коммитов) |
| shura | https://github.com/hikkian/shura | PUBLIC | MIT | `origin/main` | `b09ea523da498608575062066d5b7d5a879be9e3` |

Правило истории: берутся только коммиты, достижимые из `origin/main`. Локальная ветка `shura` у `~/Projects/shura` опережает публичную на 273 коммита; эти коммиты не читались и не используются.

Не использованы:
- `~/Projects/shura.cpp` (форк llama.cpp, C++, сборка тяжёлая; тесты не на unittest);
- `shura-integration`, `shuracode-integration`, личные наборы задач, рабочие и клиентские репозитории (их нет в списке открытых);
- чужие открытые наборы (по карточке без отдельного слова не скачиваются).

Периоды истории, которые теперь заняты экзаменом (для SELFTRAIN: не брать эти коммиты как источник практики):
- barid: все коммиты, которые вошли в `manifest-micro-v1.json` (поле `commit`, с родителем);
- shura: те же правила, если коммит попал в манифест.

## EXAM200e: открытые наборы (источник А), проверено 2026-11-14

| Набор | Адрес | Лицензия (проверена через API) | Что взято | Хеш / коммит | Статус |
|---|---|---|---|---|---|
| HumanEval | https://github.com/openai/human-eval | MIT | `data/HumanEval.jsonl.gz`, 164 задачи | sha256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` | взят |
| QuixBugs | https://github.com/jkoppel/QuixBugs | MIT (файл LICENSE) | `python_programs` (с ошибкой), `correct_python_programs` (эталон), `json_testcases`: 31 программа с тестами и эталоном, 24 прошли проверку | commit `4257f44b0ff1181dedaedee6a447e133219fcebf` | взят |
| MBPP (sanitized) | https://github.com/google-research/google-research/tree/master/mbpp | README лицензии не содержит; репозиторий Apache-2.0 в целом | скачан `sanitized-mbpp.json` (427) | sha256 `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9` | НЕ взят: лицензия набора не ясна (правило карточки) |
| BigCodeBench | https://github.com/bigcode-project/bigcodebench | Apache-2.0 (код) | в репозитории нет набора задач (он на внешнем хранилище) | — | НЕ взят: данных в репозитории нет |
| EvalPlus (исправленные HumanEval+ / MBPP+) | https://github.com/evalplus/evalplus | Apache-2.0 (код) | данные не скачаны: релиз не содержит наборов, а установка пакета запрещена | — | НЕ взят: запечатанная часть из исправленных версий не набрана |
| MultiPL-E | https://github.com/nuprl/MultiPL-E | NOASSERTION (лицензия не распознана) | — | — | НЕ взят |

Вне карточки не скачивалось. В экзамене: HumanEval 109 задач (111 прошли проверку эталоном; he-043 и he-061 сняты как дубли по эталону: в пользу he-040 и he-056) и QuixBugs 23 программы (24 прошли проверку; qb-longest_common_subsequence снят: при нескольких правильных ответах правило выбора не описано).

**Занято экзаменом (источники практики SELFTRAIN не должны брать эти задачи):**
- HumanEval: все 164 задачи набора (по id `HumanEval/0`–`HumanEval/163`);
- QuixBugs: все программы, которые вошли в манифест v2 (`qb-*`);
- EXAM200b: коммиты из `manifest-micro-v1.json` (8 задач).

**Изменения в задачах:** переименование функции (solve_*), переписывание условия по-русски. Изменения поведения не делались (0 из 140 задач), поэтому доля задач с изменённым поведением равна нулю.
