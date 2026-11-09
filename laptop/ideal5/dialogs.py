# -*- coding: utf-8 -*-
"""Dialog fixtures for prefix_check.py.

Each dialog is a dict:
  name                  short id
  chat_template_kwargs  passed to the template (jinja2 globals / engine chat_template_kwargs)
  tools                 OpenAI tools list, or None
  messages              OpenAI-style messages (system, user, assistant[with reasoning_content / tool_calls], tool)

Every assistant message is followed by a user message or by one or more tool messages,
and every dialog ends with a user message, so each assistant turn has a "k+1" to compare.
Tool-call arguments are JSON objects (not strings), as in the task brief.
"""

SYSTEM = ("You are ShuraCode, a coding agent working inside the user's repository. "
          "Use the tools to read and change files, keep answers short and concrete.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the project and return numbered lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path of the file"},
            "start_line": {"type": "integer", "description": "First line to return"},
            "end_line": {"type": "integer", "description": "Last line to return"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep_code",
        "description": "Search the project for a regular expression (ripgrep syntax).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "Directory or file to search"}},
            "required": ["pattern", "path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path of the file"},
            "content": {"type": "string", "description": "Complete new file content"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the project and return stdout and stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout_s": {"type": "integer", "description": "Timeout in seconds"}},
            "required": ["command"]}}},
]


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _tool(cid, text):
    return {"role": "tool", "tool_call_id": cid, "content": text}


# ---------------------------------------------------------------- 1. RU -> EN switch, effort low
D1 = {
    "name": "ru-en-switch",
    "chat_template_kwargs": {"reasoning_effort": "low"},
    "tools": None,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "В /home/user/proj/app/services/cache.py функция get_user_cache падает с KeyError, "
                                    "если ключа нет в _store. Сделай так, чтобы она возвращала None."},
        {"role": "assistant",
         "reasoning_content": "Смотрю на get_user_cache. Она обращается к _store[key] напрямую, поэтому отсутствующий "
                              "ключ даёт KeyError. Достаточно заменить на dict.get: тогда без ключа вернётся None, "
                              "а остальная логика не меняется. Тип возврата обновлю на int | None.",
         "content": "Причина в `get_user_cache`: она делает `_store[key]` напрямую. Заменяю на `.get()`:\n"
                    "```python\ndef get_user_cache(key: str) -> int | None:\n    return _store.get(key)\n```\n"
                    "Теперь при отсутствии ключа функция возвращает `None` и не бросает исключение."},
        {"role": "user", "content": "Thanks. Now add a pytest case for the missing-key path in "
                                    "/home/user/proj/tests/test_cache.py, and check that the existing tests still pass."},
        {"role": "assistant",
         "reasoning_content": "The new test calls get_user_cache with a key that was never stored and asserts the result "
                              "is None. I keep the existing hit test so a regression in the normal path is still caught. "
                              "The check to run afterwards is pytest -q on that file.",
         "content": "Added the test to `tests/test_cache.py`:\n```python\nfrom app.services.cache import get_user_cache, _store\n\n"
                    "def test_missing_key_returns_none():\n    _store.clear()\n    assert get_user_cache(\"user:404\") is None\n```\n"
                    "Next step is `pytest -q tests/test_cache.py` to confirm the existing cases still pass."},
        {"role": "user", "content": "Also rename _store to _user_cache across the module, the tests included, "
                                    "and tell me which other files import it."},
        {"role": "assistant",
         "reasoning_content": "A rename touches app/services/cache.py and tests/test_cache.py. I still need to find the "
                              "importers, so I will search for the name before I report the list.",
         "content": "Renamed `_store` to `_user_cache` in `app/services/cache.py` and `tests/test_cache.py`. "
                    "Only `app/services/warmup.py` imports it, with `from app.services.cache import _store`, "
                    "so that import has to change as well."},
        {"role": "user", "content": "Good. Make that import change in warmup.py too and tell me when it is done."},
    ],
}

# ---------------------------------------------------------------- 2. EN -> RU switch, effort low
D2 = {
    "name": "en-ru-switch",
    "chat_template_kwargs": {"reasoning_effort": "low"},
    "tools": None,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "The pagination helper in /home/user/proj/app/api/pagination.py returns page_count=0 "
                                    "for an empty result set. It should return 1 page instead."},
        {"role": "assistant",
         "reasoning_content": "page_count uses floor division, so total=0 gives 0. A ceiling division with a floor of 1 "
                              "fixes that without changing the normal cases.",
         "content": "The fix is a ceiling division with a minimum of one page:\n```python\n"
                    "def page_count(total: int, per_page: int) -> int:\n    return max(1, -(-total // per_page))\n```\n"
                    "For `total=0` this returns 1, and for `total=45, per_page=10` it still returns 5."},
        {"role": "user", "content": "Хорошо. Теперь проверь, что при per_page=0 не будет деления на ноль, "
                                    "и добавь проверку с понятной ошибкой ValueError."},
        {"role": "assistant",
         "reasoning_content": "Нужна проверка в начале функции: per_page <= 0 должен давать ValueError до деления. "
                              "Сообщение об ошибке пишу по-русски, как и остальные сообщения в проекте.",
         "content": "Добавил проверку в `app/api/pagination.py`, до деления:\n```python\n"
                    "def page_count(total: int, per_page: int) -> int:\n"
                    "    if per_page <= 0:\n        raise ValueError(\"per_page должен быть больше нуля\")\n"
                    "    return max(1, -(-total // per_page))\n```\n"
                    "Теперь `per_page=0` сразу даёт понятную ошибку, а не ZeroDivisionError."},
        {"role": "user", "content": "Хорошо. А в тесты tests/api/test_pagination.py добавь кейсы для total=0 и per_page=0."},
        {"role": "assistant",
         "reasoning_content": "Два новых кейса: total=0 должен давать одну страницу, а per_page=0 должен поднимать "
                              "ValueError. Оба проверяю через pytest.raises и обычный assert.",
         "content": "Добавлю два кейса в `tests/api/test_pagination.py`:\n```python\nimport pytest\n"
                    "from app.api.pagination import page_count\n\n"
                    "def test_empty_result_is_one_page():\n    assert page_count(0, 20) == 1\n\n"
                    "def test_zero_per_page_raises():\n    with pytest.raises(ValueError):\n        page_count(10, 0)\n```"},
        {"role": "user", "content": "Отлично. Покажи итоговый diff для этих двух файлов."},
    ],
}

# ---------------------------------------------------------------- 3. multi-step tool calls, effort high (English)
RETRY_V1 = ("import time\nfrom functools import wraps\n\n\ndef retry(attempts=3, base_delay=0.5):\n"
            "    def deco(fn):\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n"
            "            last_exc = None\n            for _ in range(attempts):\n                try:\n"
            "                    return fn(*args, **kwargs)\n                except ConnectionError as exc:\n"
            "                    last_exc = exc\n                    time.sleep(base_delay)\n"
            "            raise last_exc\n        return wrapper\n    return deco\n")

RETRY_V2 = ("import time\nfrom functools import wraps\n\n\ndef retry(attempts=3, base_delay=0.5, sleep=time.sleep):\n"
            "    def deco(fn):\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n"
            "            last_exc = None\n            for attempt in range(attempts):\n                try:\n"
            "                    return fn(*args, **kwargs)\n                except ConnectionError as exc:\n"
            "                    last_exc = exc\n                    if attempt < attempts - 1:\n"
            "                        sleep(base_delay * 2 ** attempt)\n"
            "            raise last_exc\n        return wrapper\n    return deco\n")

TEST_V1 = ("import time\nfrom app.net.retry import retry\n\n\ndef test_backoff_grows():\n    calls = []\n"
           "    @retry(attempts=3, base_delay=0.01)\n    def flaky():\n        calls.append(time.monotonic())\n"
           "        raise ConnectionError(\"down\")\n    try:\n        flaky()\n    except ConnectionError:\n        pass\n"
           "    gaps = [b - a for a, b in zip(calls, calls[1:])]\n    assert gaps[1] > gaps[0] * 1.5\n")

TEST_V2 = ("from app.net.retry import retry\n\n\ndef test_delays_double_each_attempt():\n    delays = []\n"
           "    @retry(attempts=4, base_delay=0.5, sleep=delays.append)\n    def flaky():\n"
           "        raise ConnectionError(\"down\")\n    try:\n        flaky()\n    except ConnectionError:\n        pass\n"
           "    assert delays == [0.5, 1.0, 2.0]\n\n\ndef test_gives_up_after_attempts():\n    calls = []\n"
           "    @retry(attempts=3, base_delay=0, sleep=lambda s: None)\n    def always_fails():\n"
           "        calls.append(1)\n        raise ConnectionError(\"down\")\n    try:\n        always_fails()\n"
           "    except ConnectionError:\n        pass\n    assert len(calls) == 3\n\n\n"
           "def test_success_skips_sleep():\n    delays = []\n    @retry(attempts=3, base_delay=0.5, sleep=delays.append)\n"
           "    def ok():\n        return 42\n    assert ok() == 42\n    assert delays == []\n")

D3 = {
    "name": "tools-high-en",
    "chat_template_kwargs": {"reasoning_effort": "high"},
    "tools": TOOLS,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "The retry decorator in /home/user/proj/app/net/retry.py never backs off, and "
                                    "tests/test_retry.py is flaky on CI. Find the cause, fix it, and run the tests "
                                    "before you finish."},
        {"role": "assistant",
         "reasoning_content": "Start with the decorator and its test. I need to see both before I decide whether the "
                              "flakiness is in the code or in the timing assertion.",
         "content": "I'll read the decorator first.",
         "tool_calls": [_call("call_1", "read_file", {"path": "/home/user/proj/app/net/retry.py",
                                                      "start_line": 1, "end_line": 80})]},
        _tool("call_1", "1  import time\n2  from functools import wraps\n...\n5  def retry(attempts=3, base_delay=0.5):\n"
                        "...\n14                    time.sleep(base_delay)\n...\n"),
        {"role": "assistant",
         "reasoning_content": "The sleep is a constant base_delay on every iteration, so there is no backoff at all. Let me "
                              "check how the test measures timing before I change the decorator.",
         "content": "",
         "tool_calls": [_call("call_2", "read_file", {"path": "/home/user/proj/tests/test_retry.py"})]},
        _tool("call_2", TEST_V1),
        {"role": "assistant",
         "reasoning_content": "The test compares gaps with a ratio of 1.5. With constant sleeps the gaps are equal plus "
                              "scheduler noise, so the ratio is random. I should check other callers before I change "
                              "the signature.",
         "content": "The decorator sleeps a constant `base_delay` between attempts, and the test expects 1.5x growth "
                    "between gaps. I'll check who else uses `retry` before changing it.",
         "tool_calls": [_call("call_3", "grep_code", {"pattern": "retry\\(", "path": "/home/user/proj/app"})]},
        _tool("call_3", "app/net/retry.py:5:def retry(attempts=3, base_delay=0.5):\n"
                        "app/net/client.py:12:@retry(attempts=5, base_delay=0.2)\n"
                        "app/net/client.py:40:@retry(attempts=3)\n"
                        "app/jobs/sync.py:9:    @retry(attempts=2, base_delay=1.0)\n"),
        {"role": "assistant",
         "reasoning_content": "client.py and sync.py pass keyword arguments only, so an optional sleep parameter and an "
                              "exponential schedule will not break them. I'll write the fix and then run the test.",
         "content": "",
         "tool_calls": [_call("call_4", "write_file", {"path": "/home/user/proj/app/net/retry.py", "content": RETRY_V2})]},
        _tool("call_4", "Wrote 19 lines to /home/user/proj/app/net/retry.py"),
        {"role": "assistant",
         "reasoning_content": "Run the retry tests only, so the output stays short. If the old timing test still fails, "
                              "the assertion message will show why.",
         "content": "",
         "tool_calls": [_call("call_5", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/test_retry.py -q",
                                                       "timeout_s": 120})]},
        _tool("call_5", ".F.\n\n=== FAILURES ===\n___ test_backoff_grows ___\n"
                        "E   assert False\n\nFAILED tests/test_retry.py::test_backoff_grows\n1 failed, 2 passed in 0.31s"),
        {"role": "assistant",
         "reasoning_content": "The decorator now sleeps 0.5, 1.0, 2.0 and the new behavior is right. The old test measures "
                              "wall-clock gaps, which is the flaky part. I will rewrite it to inject a sleep recorder and "
                              "assert the exact delays.",
         "content": "",
         "tool_calls": [_call("call_6", "write_file", {"path": "/home/user/proj/tests/test_retry.py", "content": TEST_V2})]},
        _tool("call_6", "Wrote 27 lines to /home/user/proj/tests/test_retry.py"),
        {"role": "assistant",
         "reasoning_content": "Final run of the retry tests. All three should pass, and the delays are checked exactly now.",
         "content": "",
         "tool_calls": [_call("call_7", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/test_retry.py -q",
                                                       "timeout_s": 120})]},
        _tool("call_7", "...\n3 passed in 0.02s"),
        {"role": "assistant",
         "reasoning_content": "All three tests pass. Callers still use keyword arguments only. Summarise the change for the user.",
         "content": "Fixed the flaky retry test.\n\n- `app/net/retry.py`: the backoff is now `base_delay * 2 ** attempt`, "
                    "with an injectable `sleep` for tests.\n- `tests/test_retry.py`: checks the exact delays through a "
                    "recorder instead of comparing wall-clock gaps.\n\n`pytest tests/test_retry.py -q` reports 3 passed. "
                    "`app/net/client.py` and `app/jobs/sync.py` are unchanged."},
        {"role": "user", "content": "Good. Also make the jitter configurable through the RETRY_JITTER environment "
                                    "variable, default 0."},
    ],
}

# ---------------------------------------------------------------- 4. same style, effort max, user writes Russian
LOADER_V1 = ("import csv\n\n\ndef load_rows(path):\n    with open(path, encoding=\"utf-8\") as f:\n"
             "        return [row for row in csv.DictReader(f)]\n")

LOADER_V2 = ("import csv\nfrom pathlib import Path\n\nENCODINGS = (\"utf-8-sig\", \"cp1251\")\n\n\n"
             "def _read_text(path: Path) -> str:\n    raw = path.read_bytes()\n    for enc in ENCODINGS:\n"
             "        try:\n            return raw.decode(enc)\n        except UnicodeDecodeError:\n            continue\n"
             "    raise ValueError(f\"{path}: неизвестная кодировка\")\n\n\n"
             "def load_rows(path: str | Path) -> list[dict[str, str]]:\n    text = _read_text(Path(path))\n"
             "    reader = csv.DictReader(text.splitlines())\n    return [row for row in reader if any(row.values())]\n")

D4 = {
    "name": "tools-max-ru",
    "chat_template_kwargs": {"reasoning_effort": "max"},
    "tools": TOOLS,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "В /home/user/proj/app/parsers/csv_loader.py функция load_rows падает с "
                                    "UnicodeDecodeError на файлах из Excel: там BOM и кодировка cp1251. Нужно, чтобы "
                                    "файл читался и так, и так, а tests/parsers/test_csv_loader.py проходил."},
        {"role": "assistant",
         "reasoning_content": "Сначала посмотрю текущий загрузчик. Ошибка говорит о том, что файл читается только как UTF-8, "
                              "без учёта BOM и cp1251. Нужно понять, как функция используется, прежде чем менять её.",
         "content": "",
         "tool_calls": [_call("call_1", "read_file", {"path": "/home/user/proj/app/parsers/csv_loader.py"})]},
        _tool("call_1", LOADER_V1),
        {"role": "assistant",
         "reasoning_content": "Причина в open(path, encoding='utf-8'): BOM от Excel остаётся в первом заголовке, а байты cp1251 "
                              "не декодируются как UTF-8. Запускаю тесты парсера, чтобы увидеть точные падения.",
         "content": "",
         "tool_calls": [_call("call_2", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/parsers -q",
                                                       "timeout_s": 120})]},
        _tool("call_2", "F....\n=== FAILURES ===\nE   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xcf in "
                        "position 0: invalid continuation byte\nFAILED tests/parsers/test_csv_loader.py::test_cp1251_file\n"
                        "1 failed, 4 passed in 0.44s"),
        {"role": "assistant",
         "reasoning_content": "Нужны две кодировки подряд: сначала utf-8-sig, который снимает BOM, затем cp1251 для файлов "
                              "Excel без UTF-8. Пустые строки из хвоста файла тоже надо отбрасывать. Интерфейс "
                              "load_rows оставляю тем же, чтобы вызывающий код не менялся.",
         "content": "",
         "tool_calls": [_call("call_3", "write_file", {"path": "/home/user/proj/app/parsers/csv_loader.py", "content": LOADER_V2})]},
        _tool("call_3", "Wrote 22 lines to /home/user/proj/app/parsers/csv_loader.py"),
        {"role": "assistant",
         "reasoning_content": "Запускаю тесты парсера ещё раз: нужно убедиться, что все пять проходят, включая cp1251.",
         "content": "",
         "tool_calls": [_call("call_4", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/parsers -q",
                                                       "timeout_s": 120})]},
        _tool("call_4", ".....\n5 passed in 0.41s"),
        {"role": "assistant",
         "reasoning_content": "Все пять тестов проходят. Проверю формулировку итога: что изменено, какие кодировки поддержаны, "
                              "что не трогал.",
         "content": "Исправил `app/parsers/csv_loader.py`: файл теперь читается сначала как `utf-8-sig` (BOM из Excel "
                    "снимается), а если не получилось, то как `cp1251`. Пустые строки в конце файла пропускаются.\n\n"
                    "Интерфейс `load_rows(path)` не менялся. Все 5 тестов в `tests/parsers` проходят, включая `test_cp1251_file`."},
        {"role": "user", "content": "Отлично. А для TSV-файлов тоже так работает? Проверь на примере и скажи, "
                                    "нужен ли тут отдельный разделитель."},
    ],
}

# ---------------------------------------------------------------- 5. enable_thinking=false
D5 = {
    "name": "thinking-off-en",
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": None,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Every time I change a line in /home/user/proj/app/main.py, docker build reinstalls "
                                    "all pip packages. Why does that happen, and how do I fix the Dockerfile?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "Docker caches each layer, and a `COPY . .` before `pip install` invalidates the cache for every "
                    "source change. Copy the requirements file first:\n```\nFROM python:3.12-slim\nWORKDIR /srv/app\n"
                    "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\n"
                    "CMD [\"python\", \"-m\", \"app.main\"]\n```\nNow the install layer is reused until requirements.txt changes."},
        {"role": "user", "content": "It works now. What about .dockerignore? Should I exclude the .venv folder in /home/user/proj?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "Yes. A local `.venv` copied into the image is large and can shadow the installed packages. Add this to "
                    "`/home/user/proj/.dockerignore`:\n```\n.venv/\n__pycache__/\n.git/\n*.pyc\n```"},
        {"role": "user", "content": "And how do I pin the base image so the build is reproducible?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "Pin the image by digest. Run `docker buildx imagetools inspect python:3.12-slim`, copy the `Digest` "
                    "line, and write `FROM python:3.12-slim@sha256:<that digest>` in the Dockerfile. Re-check the digest "
                    "when you want to move to a newer base image."},
        {"role": "user", "content": "Thanks, that's enough for today."},
    ],
}

# ---------------------------------------------------------------- 6. <context> block + Russian request, effort xhigh, tools
XLSX_V1 = ("from openpyxl import Workbook\n\nHEADERS = [\"id\", \"name\", \"created_on\"]\n\n\n"
           "def write_rows(path, rows):\n    wb = Workbook()\n    ws = wb.active\n    ws.append(HEADERS)\n"
           "    for row in rows:\n        created = row[\"created_on\"]\n"
           "        ws.append([row[\"id\"], row[\"name\"], f\"{created:%Y-%m-%d}\"])\n    wb.save(path)\n")

XLSX_V2 = ("from openpyxl import Workbook\n\nHEADERS = [\"id\", \"name\", \"created_on\"]\n\n\n"
           "def _format_date(value):\n    return \"\" if value is None else f\"{value:%Y-%m-%d}\"\n\n\n"
           "def write_rows(path, rows):\n    wb = Workbook()\n    ws = wb.active\n    ws.append(HEADERS)\n"
           "    for row in rows:\n        ws.append([row[\"id\"], row[\"name\"], _format_date(row[\"created_on\"])])\n"
           "    wb.save(path)\n")

D6 = {
    "name": "context-xhigh-ru",
    "chat_template_kwargs": {"reasoning_effort": "xhigh"},
    "tools": TOOLS,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "<context>\nWorking directory: /home/user/proj\nGit branch: feature/export-xlsx\n"
                                    "Open files: app/export/xlsx_writer.py\n</context>\n\n"
                                    "Сделай так, чтобы экспорт в Excel не падал с TypeError, когда в поле даты стоит None, "
                                    "и пустые даты выводились пустой ячейкой."},
        {"role": "assistant",
         "reasoning_content": "Сначала открою xlsx_writer.py: нужно понять, как пишутся даты и почему None даёт TypeError. "
                              "Тесты экспорта посмотрю после этого, чтобы знать текущие ожидания.",
         "content": "",
         "tool_calls": [_call("call_1", "read_file", {"path": "/home/user/proj/app/export/xlsx_writer.py"})]},
        _tool("call_1", XLSX_V1),
        {"role": "assistant",
         "reasoning_content": "Ошибка в f\"{created:%Y-%m-%d}\": для None форматирование с непустым спецификатором даёт "
                              "TypeError. Нужна пустая строка для None. Сначала запущу тесты экспорта, чтобы увидеть, "
                              "какие именно падают.",
         "content": "",
         "tool_calls": [_call("call_2", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/export -q",
                                                       "timeout_s": 120})]},
        _tool("call_2", "..F..\n=== FAILURES ===\nE   TypeError: unsupported format string passed to NoneType.__format__\n"
                        "FAILED tests/export/test_xlsx_writer.py::test_none_date_is_empty_cell\n"
                        "1 failed, 4 passed in 0.58s"),
        {"role": "assistant",
         "reasoning_content": "Есть ли такая же строка в других экспортёрах? Посмотрю весь модуль export, прежде чем менять функцию.",
         "content": "",
         "tool_calls": [_call("call_3", "grep_code", {"pattern": "%Y-%m-%d", "path": "/home/user/proj/app/export"})]},
        _tool("call_3", "app/export/xlsx_writer.py:9:        ws.append([row[\"id\"], row[\"name\"], f\"{created:%Y-%m-%d}\"])\n"
                        "app/export/csv_writer.py:14:        out.write(f\"{row['created_on']:%Y-%m-%d}\\n\")\n"),
        {"role": "assistant",
         "reasoning_content": "Нашлась та же строка в csv_writer.py. Пользователь попросил только Excel, поэтому csv не трогаю, "
                              "а упомяну его в ответе. В xlsx_writer.py выношу форматирование даты в отдельную функцию, "
                              "которая возвращает пустую строку для None.",
         "content": "",
         "tool_calls": [_call("call_4", "write_file", {"path": "/home/user/proj/app/export/xlsx_writer.py", "content": XLSX_V2})]},
        _tool("call_4", "Wrote 17 lines to /home/user/proj/app/export/xlsx_writer.py"),
        {"role": "assistant",
         "reasoning_content": "Запускаю тесты экспорта ещё раз, чтобы убедиться, что все пять проходят, включая тест на пустую дату.",
         "content": "",
         "tool_calls": [_call("call_5", "run_command", {"command": "cd /home/user/proj && python -m pytest tests/export -q",
                                                       "timeout_s": 120})]},
        _tool("call_5", ".....\n5 passed in 0.57s"),
        {"role": "assistant",
         "reasoning_content": "Все пять тестов проходят. В ответе перечислю, что изменено, и отдельно отмечу csv_writer.py с той же ошибкой.",
         "content": "Исправил `app/export/xlsx_writer.py`: пустая дата теперь пишется пустой ячейкой через функцию "
                    "`_format_date`. Тесты в `tests/export` проходят (5 passed).\n\n"
                    "В `app/export/csv_writer.py` строка 14 содержит ту же ошибку (`f\"{row['created_on']:%Y-%m-%d}\"`). "
                    "Я её не менял, потому что вы просили только Excel."},
        {"role": "user", "content": "Спасибо. Тогда поправь и csv_writer.py тоже и добавь тест на пустую дату для CSV."},
    ],
}

# ---------------------------------------------------------------- 7. long mixed chat: KZ, EN, RU, EN; effort low; reasoning
D7 = {
    "name": "long-mixed-kz-en-ru",
    "chat_template_kwargs": {"reasoning_effort": "low"},
    "tools": None,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Менің slugify функциям /home/user/proj/app/utils/slug.py файлында кириллицаны "
                                    "өшіріп жібереді. Түзетуге көмектесе аласың ба?"},
        {"role": "assistant",
         "reasoning_content": "The slugify regex keeps only [a-z0-9], so every Cyrillic letter is dropped. The fix is to use "
                              "the Unicode-aware \\w class. The user writes in Kazakh, so the answer must be in Kazakh too.",
         "content": "Иә, көмектесе аламын. Себебі `slugify` функциясындағы регулярлық өрнек `[^a-z0-9]` кириллица әріптерін "
                    "бірден өшіріп тастайды. Оның орнына Unicode-ке сүйенетін `\\w` класын қолдануға болады:\n```python\n"
                    "import re\n\ndef slugify(text: str) -> str:\n    text = text.lower().strip()\n"
                    "    text = re.sub(r\"[^\\w\\s-]\", \"\", text)\n    return re.sub(r\"[\\s_]+\", \"-\", text).strip(\"-\")\n```"},
        {"role": "user", "content": "Thanks. Does \\w handle Kazakh letters like ә and ң in the new version? "
                                    "Show a quick check in the REPL."},
        {"role": "assistant",
         "reasoning_content": "Python 3 re patterns are Unicode-aware for str, so \\w matches Kazakh letters. A REPL check with "
                              "a mixed Kazakh and Latin title is enough to show it.",
         "content": "Yes. In Python 3 `re` patterns are Unicode-aware for `str`, so `\\w` matches Cyrillic letters, including "
                    "ә, ғ, қ, ң, ө, ұ, ү, һ and і. A quick check:\n```\n>>> from app.utils.slug import slugify\n"
                    ">>> slugify('Қазақ тілі: Әлем & ұлттар')\n'қазақ-тілі-әлем-ұлттар'\n```\n"
                    "The colon and the ampersand are removed, and the spaces become hyphens."},
        {"role": "user", "content": "Понятно. Теперь проверь, как работает этот slugify с пустой строкой и со строкой только "
                                    "из знаков препинания. Ответь по-русски."},
        {"role": "assistant",
         "reasoning_content": "For an empty string the regex finds nothing and strip('-') returns an empty string. A string "
                              "of punctuation only also gives an empty string. The caller should fall back to a default, "
                              "so I should show that too. The answer is in Russian.",
         "content": "Для пустой строки результат будет `''`: регулярные выражения ничего не находят. Для строки из одних "
                    "знаков препинания, например `'!!! ???'`, тоже будет `''`. В вызывающем коде стоит это учесть, например так:\n"
                    "```python\nslug = slugify(title) or 'untitled'\n```"},
        {"role": "user", "content": "Хорошо. Добавь этот случай в tests/utils/test_slug.py, пожалуйста, и покажи, "
                                    "какие тесты там уже есть."},
        {"role": "assistant",
         "reasoning_content": "The test file is not in this conversation, so I cannot list its existing tests. I can give the "
                              "new test case and say that clearly. Russian answer.",
         "content": "Вот тест, который стоит добавить в `tests/utils/test_slug.py`:\n```python\n"
                    "def test_empty_and_punctuation_only():\n    assert slugify('') == ''\n    assert slugify('!!! ???') == ''\n```\n"
                    "Существующие тесты в этом разговоре я не видел, поэтому их не перечисляю."},
        {"role": "user", "content": "Got it. One last thing: summarise in English what changed in slug.py."},
    ],
}

DIALOGS = [D1, D2, D3, D4, D5, D6, D7]
