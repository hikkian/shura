"""Post-install smoke checks. Each check returns (name, status, text); status is ok / fail / not_checked.
run_smoke(base_url, profile) -> {"results": [...], "summary": "ok"|"fail"|"not_checked", "text": ...}; it never raises.
Only the standard library; talks to base_url over HTTP, reads /proc/meminfo and (if present) amdgpu sysfs.

profile keys (all optional): exe_path, expected_model, expected_mode, health_timeout_s (60), request_timeout_s (60),
min_free_vram_mib, min_free_ram_mib, pause_path ("/pause"), resume_path ("/resume"), vram_path ("/vram").
run_smoke(..., sources={"vram_free_mib": callable, "ram_free_mib": callable}) replaces the memory readers (tests)."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

OK, FAIL, NOT_CHECKED = "ok", "fail", "not_checked"


def _http(base, path, body=None, timeout=10, method=None):
    """-> (status, parsed json or None, raw text). Raises OSError/URLError on connection problems."""
    parts = urllib.parse.urlsplit(str(base))
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"smoke test talks only to http://127.0.0.1 or http://localhost, not {base!r}")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read().decode("utf-8", "replace")
    try:
        return code, json.loads(raw), raw
    except ValueError:
        return code, None, raw


def _ram_free_mib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _vram_free_mib():
    import glob
    best = None
    for d in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            total = int(open(d + "/mem_info_vram_total").read())
            used = int(open(d + "/mem_info_vram_used").read())
        except (OSError, ValueError):
            continue
        if best is None or total > best[0]:
            best = (total, used)
    return None if best is None else (best[0] - best[1]) / 1048576.0


def check_health(base, p):
    name = "server_up"
    limit = float(p.get("health_timeout_s", 60))
    deadline, last = time.time() + limit, "no answer"
    while True:
        try:
            code, js, _ = _http(base, "/health", timeout=min(5, limit))
            if code == 200:
                break
            last = "HTTP %s" % code
        except Exception as e:  # noqa: BLE001
            last = str(e)
        if time.time() >= deadline:
            return (name, FAIL, "сервер не ответил 200 на /health за %g с (%s): проверьте журнал службы ai-gateway и путь к бинарнику exePath%s"
                    % (limit, last, " (%s)" % p["exe_path"] if p.get("exe_path") else ""))
        time.sleep(0.2)
    seen = json.dumps(js, ensure_ascii=False) if js is not None else ""
    for key in ("expected_model", "expected_mode"):
        want = p.get(key)
        if want and js is not None:
            field = key.split("_", 1)[1]
            got = js.get(field)
            if got is not None and str(got) != str(want):
                return (name, FAIL, "сервер поднялся, но %s=%r, ожидалось %r: проверьте выбранный режим/модель в профиле и перезапустите службу" % (field, got, want))
    return (name, OK, "health 200 %s" % seen[:120])


def _chat(base, p, extra=None):
    body = {"messages": [{"role": "user", "content": "Say hi in one short sentence."}], "max_tokens": 32, "stream": False}
    body.update(extra or {})
    return _http(base, "/v1/chat/completions", body, timeout=float(p.get("request_timeout_s", 60)))


def check_first_reply(base, p, cache):
    name = "first_reply"
    try:
        code, js, raw = _chat(base, p)
    except Exception as e:  # noqa: BLE001
        return (name, FAIL, "запрос к /v1/chat/completions не удался (%s): проверьте, что модель загружена, и журнал службы ai-gateway" % e)
    cache["chat"] = js
    if code != 200 or not isinstance(js, dict):
        return (name, FAIL, "первый запрос вернул HTTP %s: %s. Проверьте журнал службы ai-gateway" % (code, raw[:150]))
    try:
        ch = js["choices"][0]
        msg = ch.get("message") or {}
        text = (msg.get("content") or "").strip()
        reason = ch.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return (name, FAIL, "ответ без choices: сервер совместим с OpenAI API? Проверьте версию бинарника (exePath)")
    if not text:
        return (name, FAIL, "ответ пустой (finish_reason=%s): модель загрузилась, но ничего не сказала. Проверьте шаблон чата и файл модели" % reason)
    if not reason:
        return (name, FAIL, "в ответе нет finish_reason: ответ оборвался. Проверьте журнал службы ai-gateway")
    return (name, OK, "ответ %r, finish_reason=%s" % (text[:40], reason))


def check_mtp(base, p, cache):
    name = "mtp_accepts"
    js = cache.get("chat")
    t = (js or {}).get("timings") if isinstance(js, dict) else None
    t = t if isinstance(t, dict) else {}
    drafted = t.get("draft_n", t.get("draft_tokens"))
    accepted = t.get("draft_n_accepted", t.get("draft_accepted"))
    if drafted is None or accepted is None:
        return (name, NOT_CHECKED, "сервер не отдаёт счётчики draft/accepted: приём токенов MTP не проверен")
    if drafted > 0 and accepted <= 0:
        return (name, FAIL, "MTP предложил %s токенов, принято 0: проверьте, что MTP-голова подключена к модели и включена в режиме" % drafted)
    if drafted == 0:
        return (name, NOT_CHECKED, "в этом запросе MTP ничего не предлагал (draft=0): приём не проверен")
    return (name, OK, "MTP принято %s из %s" % (accepted, drafted))


def check_memory(base, p, sources):
    name = "memory_bounds"
    problems, seen, unknown = [], [], []
    for label, key, src_key, reader in (("VRAM", "min_free_vram_mib", "vram_free_mib", _vram_free_mib),
                                        ("ОЗУ", "min_free_ram_mib", "ram_free_mib", _ram_free_mib)):
        bound = p.get(key)
        if bound is None:
            unknown.append("%s: в профиле нет границы" % label)
            continue
        try:
            free = (sources.get(src_key) or reader)()
        except Exception:  # noqa: BLE001
            free = None
        if free is None:
            unknown.append("%s: нет источника значения" % label)
            continue
        seen.append("%s свободно %.0f МиБ (граница %s)" % (label, free, bound))
        if free < float(bound):
            problems.append("%s: свободно меньше границы (%.0f < %s МиБ): закройте тяжёлые программы или выберите режим с меньшим контекстом" % (label, free, bound))
    if problems:
        return (name, FAIL, "; ".join(problems))
    if not seen:
        return (name, NOT_CHECKED, "границы памяти не проверены (" + "; ".join(unknown) + ")")
    return (name, OK, "; ".join(seen + unknown))


def check_guardian(base, p):
    name = "guardian_vram"
    try:
        code, js, _ = _http(base, p.get("vram_path", "/vram"), timeout=10)
    except Exception as e:  # noqa: BLE001
        return (name, FAIL, "сторож не ответил на /vram (%s): проверьте службу ai-gateway" % e)
    if code in (404, 405, 501):
        return (name, NOT_CHECKED, "эндпоинт /vram отсутствует (HTTP %s): у этого сервера нет сторожа, проверка пропущена" % code)
    if code != 200 or not isinstance(js, dict):
        return (name, FAIL, "/vram вернул HTTP %s без JSON: проверьте журнал службы ai-gateway" % code)
    return (name, OK, "сторож отдаёт память: " + json.dumps(js, ensure_ascii=False)[:120])


def check_pause(base, p):
    name = "pause_resume"
    steps = []
    for path in (p.get("pause_path", "/pause"), p.get("resume_path", "/resume")):
        try:
            code, _, _ = _http(base, path, {}, timeout=15, method="POST")
        except Exception as e:  # noqa: BLE001
            return (name, FAIL, "%s не удался (%s): проверьте службу ai-gateway" % (path, e))
        if code in (404, 405, 501):
            if not steps:
                return (name, NOT_CHECKED, "эндпоинта паузы %s нет (HTTP %s): пауза/возобновление не проверены" % (path, code))
            return (name, FAIL, "пауза сработала, но %s недоступен (HTTP %s): сервер остался на паузе, перезапустите службу ai-gateway" % (path, code))
        if code != 200:
            return (name, FAIL, "%s вернул HTTP %s: проверьте журнал службы ai-gateway" % (path, code))
        steps.append(path)
    try:
        code, _, _ = _http(base, "/health", timeout=10)
    except Exception as e:  # noqa: BLE001
        return (name, FAIL, "после возобновления сервер не отвечает (%s): перезапустите службу ai-gateway" % e)
    if code != 200:
        return (name, FAIL, "после возобновления /health вернул %s: перезапустите службу ai-gateway" % code)
    return (name, OK, "пауза и возобновление прошли, health 200")


def check_agent_step(base, p):
    name = "agent_step"
    tools = [{"type": "function", "function": {"name": "get_time", "description": "Return the current time.",
                                               "parameters": {"type": "object", "properties": {}}}}]
    try:
        code, js, raw = _chat(base, p, {"messages": [{"role": "user", "content": "Call the get_time tool."}],
                                        "tools": tools, "tool_choice": "auto", "max_tokens": 64})
    except Exception as e:  # noqa: BLE001
        return (name, FAIL, "запрос с инструментом не удался (%s): проверьте журнал службы ai-gateway" % e)
    if code != 200 or not isinstance(js, dict):
        return (name, NOT_CHECKED, "сервер не принял запрос с инструментами (HTTP %s): шаг агента не проверен" % code)
    try:
        calls = js["choices"][0]["message"].get("tool_calls")
    except (KeyError, IndexError, TypeError, AttributeError):
        calls = None
    if not calls:
        return (name, NOT_CHECKED, "сервер ответил без вызова инструмента (поддельный или без tool calling): шаг агента не проверен")
    try:
        json.loads(calls[0]["function"].get("arguments") or "{}")
    except (KeyError, TypeError, ValueError, AttributeError):
        return (name, FAIL, "вызов инструмента пришёл с битыми аргументами: проверьте шаблон чата (tool calling) модели")
    return (name, OK, "модель вызвала инструмент %s" % calls[0]["function"].get("name"))


def run_smoke(base_url, profile=None, sources=None):
    p, sources, results, cache = dict(profile or {}), dict(sources or {}), [], {}

    def run(fn, *a):
        try:
            results.append(fn(*a))
        except Exception as e:  # noqa: BLE001
            results.append((fn.__name__, FAIL, "внутренняя ошибка проверки: %s: %s" % (type(e).__name__, e)))

    run(check_health, base_url, p)
    if results[0][1] == FAIL:
        for nm in ("first_reply", "mtp_accepts", "guardian_vram", "pause_resume", "agent_step"):
            results.append((nm, NOT_CHECKED, "пропущено: сервер не поднялся"))
        run(check_memory, base_url, p, sources)
    else:
        run(check_first_reply, base_url, p, cache)
        run(check_mtp, base_url, p, cache)
        run(check_memory, base_url, p, sources)
        run(check_guardian, base_url, p)
        run(check_pause, base_url, p)
        run(check_agent_step, base_url, p)
    bad = [r for r in results if r[1] == FAIL]
    nc = [r for r in results if r[1] == NOT_CHECKED]
    summary = FAIL if bad else (NOT_CHECKED if nc else OK)
    text = ("провалено: " + ", ".join(r[0] for r in bad)) if bad else ("всё проверено" if not nc else "ошибок нет, не проверено: " + ", ".join(r[0] for r in nc))
    return {"results": results, "summary": summary, "text": text}
