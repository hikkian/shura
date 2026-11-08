# Материалы с ноутбука для агента домашнего ПК

Главный документ: [REPORT-home-pc.md](REPORT-home-pc.md). Начать с раздела 7.0 (почему шаблоны важнее всего) и 7.0.1 (порядок работы).

| Папка | Что внутри |
|---|---|
| `templates/` | готовые шаблоны: `*-ideal` (база 7.11), `*-ideal4` (effort v4/v5, язык ответа; в тесте на ноутбуке) |
| `template-gen/` | генератор `make_templates.py` (править его, а не .jinja), разбор `DESIGN.md`, системный промпт агента `prompt-v3.txt` (правило честности) |
| `gateway/` | `lang_guard.py` (страховка языка ответа) и diff `ai_gateway.py` относительно `main` (страховка, effort-бюджеты, `/health` без загрузки модели) |
| `config/` | запись модели `ornith4` для `model-launch.json` и `shuracode.json`; diff обёртки `shuracode` (лимит вывода за шаг 65536) |
| `bench/stand/` | стенд: `bench.sh`, `score2.py` (с исправлением pytest), `rescore.py`, `batch.sh`. Пути внутри — ноутбучные, поправить |
| `bench/tasks/` | 19 заданий: `prompt.txt`, `repo/`, `hidden_test.py` (агент под тестом не должен их видеть) |
| `bench/tools/` | `probe.py` (опечатки), `vision.py` (макет), `speed.py`, `margin.py` (квант vs модель), `merge.py` (смешивание весов, не проверен) |

Папка обновляется коммитами с ноутбука по мере новых результатов.
