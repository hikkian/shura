"""The card "what was chosen and why" (INSTALL1c, step 6): plain Russian text for a person, built from a Choice of
autofit_api.choose() and the machine profile.

    card_text(machine_name, hw, choice, *, rejected_note=None) -> str

The card tells: what machine this is (no user name, no computer name: only the label the caller gives and the parts of
the profile), which mode was chosen (model, quant, backend, window, kind of mode), what limited the choice, the expected
speed as a range marked "прогноз, не измерено", the rejected alternative with numbers, what the next hardware step would
give, how to roll back and one command to change the model later (the same path: consent, autofit, checks, rollback).
Nothing here reads the disk, the network or the environment; the same input gives the same text.
"""
import re

from . import planner

GiB = 1024 ** 3

BACKEND_NAMES = {"hip": "ROCm (HIP)", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal", "cpu": "процессор (без видеокарты)"}
MODE_NAMES = {
    "gpu": "всё в видеопамяти",
    "hybrid": "гибридный: основная часть модели в видеопамяти, остальное в ОЗУ",
    "cpu": "только процессор и ОЗУ",
    "unified": "общая память процессора и видеоядра",
}
MODE_SHORT = {"gpu": "всё в видеопамяти", "hybrid": "гибридный", "cpu": "только процессор", "unified": "общая память"}
RUNG_NAMES = {"comfort": "комфортно", "target": "цель достигнута", "none": "ниже минимума"}
LIMIT_NAMES = {"vram": "видеопамять", "ram": "оперативная память (ОЗУ)", "disk": "место на диске", "speed": "скорость"}
VENDORS = {"amd": "AMD", "nvidia": "NVIDIA", "intel": "Intel", "apple": "Apple"}
ROLLBACK = "shura uninstall"
TARGET_TOK_S = planner.DEFAULTS["target_tok_s"]
COMFORT_TOK_S = planner.DEFAULTS["comfort_tok_s"]
MIN_TOK_S = planner.DEFAULTS["min_tok_s"]
KV_EXPLAIN = {      # the kinds of context cache, in plain words (one line each)
    "turbo3": "turbo3: очень сильное сжатие кэша контекста (около 3 бит на значение вместо 16); занимает мало видеопамяти, ответы почти те же.",
    "q4_0": "q4_0: сжатие кэша контекста до 4 бит на значение (вчетверо легче несжатого); память экономится, качество чуть ниже.",
    "q8_0": "q8_0: сжатие кэша контекста до 8 бит на значение (вдвое легче несжатого); качество практически не страдает.",
}

_TRANSLATE = (        # planner phrases that are English -> Russian (the card is for a person who reads Russian)
    ("the target speed is out of reach with this window; the speed is still comfortable",
     "целевая скорость при таком окне недостижима, но скорость всё равно комфортная"),
    ("nothing reaches the minimum speed; this is the fastest that fits",
     "ни один вариант не дотягивает до минимальной скорости; это самый быстрый из тех, что помещаются"),
    ("reaches the target speed with a smaller quant than the tested default",
     "целевая скорость достигнута с более лёгким квантом, чем проверенный по умолчанию"),
    ("reaches the target speed", "целевая скорость достигнута"),
)


def _gib(n):
    return f"{n / GiB:.0f}" if n >= 10 * GiB else f"{n / GiB:.1f}"


def _kv_cache(choice):
    for r in choice.reasons:
        m = re.search(r"(\S+) KV cache", r)
        if m and r.startswith("mode:"):
            return m.group(1)
    return None


def _translate(line):
    for en, ru in _TRANSLATE:
        if en in line:
            tail = line.split(en, 1)[1]
            tail = re.sub(r"^\s*\((.*)\)\s*$", lambda m: " (не помещается или слишком медленный здесь)"
                          if "does not fit or is too slow" in m.group(1) else "", tail)
            return ru + tail
    m = re.match(r"predicted (\d+) tok/s is below the (\d+) tok/s minimum", line)
    if m:
        return f"прогноз около {m.group(1)} tok/s ниже минимума {m.group(2)} tok/s; это лучшее, что доступно на этой машине"
    return None     # an unknown English line is left out rather than shown half-understood


def _machine_lines(hw):
    lines = []
    cpu = hw.get("cpu") or {}
    if cpu.get("model"):
        lines.append(f"процессор {cpu['model']} ({cpu.get('physical_cores', '?')} ядер, {cpu.get('logical_cores', '?')} потоков)")
    mem = hw.get("memory") or {}
    if mem.get("total"):
        lines.append(f"ОЗУ {_gib(mem['total'])} ГиБ, из них доступно {_gib(mem.get('available') or mem['total'])} ГиБ")
    gpus = hw.get("gpus") or []
    if gpus:
        for g in gpus:
            lines.append(f"видеокарта {g.get('name', '?')} ({VENDORS.get(g.get('vendor'), g.get('vendor', '?'))}), "
                         f"видеопамяти {_gib(g.get('vram_total', 0))} ГиБ")
    else:
        lines.append("видеокарты, пригодной для модели, не найдено (работа только на процессоре)")
    for g in (hw.get("system") or {}).get("gpus_all") or []:
        if g.get("vendor") == "intel" and g.get("display"):
            lines.append("видеоядро Intel рисует экран; модель на нём не считается")
    free = (hw.get("disk") or {}).get("free")
    if free:
        lines.append(f"свободно на диске {_gib(free)} ГиБ")
    return lines


def _next_step(choice):
    lim = choice.limiting
    if lim == "vram":
        return "больше видеопамяти (видеокарта с большим объёмом): в видеокарту поместится больше слоёв модели и окна."
    if lim == "ram":
        return "больше ОЗУ: часть модели, не поместившаяся в видеопамять, будет жить в памяти без нехватки."
    if lim == "disk":
        return "больше свободного места на диске (или быстрый диск): тогда подойдёт более точный квант."
    if lim == "speed":
        if choice.mode == "cpu":
            return ("видеокарта с достаточной видеопамятью: на процессоре скорость упирается в скорость ОЗУ, "
                    "видеокарта сильно ускорит генерацию.")
        return "больше видеопамяти и более быстрая память: больше слоёв модели считается на видеокарте, скорость выше."
    return "по этому выбору явного узкого места нет; железо менять незачем."


def _change_commands(choice):
    """(the form with the present values, the same with the rejected alternative's model/quant or None when it is the same)."""
    now = f"shura install --model-id {choice.model_id} --quant {choice.quant}"
    alt = choice.alternative or {}
    other = f"shura install --model-id {alt['model_id']} --quant {alt['quant']}" if alt else None
    return now, (other if other and other != now else None)


def card_text(machine_name, hw, choice, *, rejected_note=None):
    out = [f"# Карточка выбора: {machine_name}", ""]
    out += ["## Что за машина", ""] + [f"- {x}" for x in _machine_lines(hw)] + [""]
    if not choice.ok:
        out += ["## Что выбрано", "",
                "Подходящего режима не нашлось: установка ничего не скачает и ничего не изменит.", "",
                "## Что ограничило выбор", ""] + [f"- {r}" for r in choice.reasons] + [""]
        if rejected_note:
            out += [rejected_note, ""]
        out += ["## Как откатить", "", f"Ничего не установлено, откатывать нечего. Если что-то осталось: `{ROLLBACK}`.", ""]
        return "\n".join(out)

    kv = _kv_cache(choice)
    out += ["## Что выбрано", "",
            f"- модель: {choice.model_id}",
            f"- квант (степень сжатия весов): {choice.quant}",
            f"- бэкенд (чем считается): {BACKEND_NAMES.get(choice.backend, choice.backend)}",
            f"- окно контекста: {choice.context} токенов" + (f", кэш контекста {kv}" if kv else ""),
            f"- вид режима: {MODE_NAMES.get(choice.mode, choice.mode)}", ""]
    if kv in KV_EXPLAIN:
        out += [f"Что такое {KV_EXPLAIN[kv]}", ""]

    out += ["## Что ограничило выбор", "", f"Главное ограничение: {LIMIT_NAMES.get(choice.limiting, 'не определено')}.", ""]
    order = [BACKEND_NAMES.get(b, b) for b in (choice.backend_order or [])]
    if order:
        out += ["Порядок бэкендов на этой машине: " + ", затем ".join(order) + ". Берётся первый, на котором модель помещается; "
                "процессор всегда последний.", ""]
    for r in choice.reasons:
        if r.startswith("ограничивает:") or r.startswith("mode:"):
            continue
        ru = r if re.search(r"[а-яА-Я]", r) else _translate(r)
        if ru:
            ru = re.sub(r"\((comfort|target|none)\)", lambda m: f"({RUNG_NAMES[m.group(1)]})", ru)
            out.append(f"- {ru}")
    for n in choice.notes:
        ru = n if re.search(r"[а-яА-Я]", n) else _translate(n)
        if ru:
            out.append(f"- {ru}")
    out.append("")

    s = choice.speed_range
    out += ["## Ожидаемая скорость", ""]
    if s.get("low") is not None:
        status = "измерено на этой машине" if s.get("measured") else "прогноз, не измерено"
        out.append(f"От {s['low']} до {s['high']} токенов в секунду (tok/s), ожидаемо около {s['mid']} ({status}).")
    else:
        out.append("Скорость не измерена и не оценена.")
    out.append(f"Что значат слова «цель» и «комфортно»: цель - {TARGET_TOK_S:.0f} tok/s (к ней стремится подбор), "
               f"комфортно - от {COMFORT_TOK_S:.0f} tok/s (ответ читается без ожидания); ниже {MIN_TOK_S:.0f} tok/s модель не рекомендуется.")
    if s.get("mid") is not None and s["mid"] < MIN_TOK_S:
        out.append(f"ВНИМАНИЕ: ожидаемая скорость ниже {MIN_TOK_S:.0f} tok/s. Ответы будут идти очень медленно, "
                   "работать с такой скоростью неудобно; это лучшее, что есть на этой машине.")
    out.append("")

    out += ["## Отвергнутая альтернатива", ""]
    a = choice.alternative
    if a:
        tok = f"около {a['tok_s']} tok/s" if a.get("tok_s") is not None else "скорость неизвестна"
        mode = f", режим {MODE_SHORT.get(a['mode'], a['mode'])}" if a.get("mode") else ""
        out.append(f"{a['model_id']}, квант {a['quant']}, бэкенд {BACKEND_NAMES.get(a['backend'], a['backend'])}, "
                   f"окно {a['context']} токенов{mode}: {tok}.")
        out.append(f"Почему не выбрана: {a['why_not']}.")
        if "счёт планировщика" in a["why_not"]:
            out.append("«Счёт планировщика» - число, по которому подбор сравнивает варианты: оно растёт с качеством модели, "
                       "размером окна и скоростью; побеждает вариант с большим числом.")
    else:
        out.append("Альтернативы нет: другой бэкенд, квант или модель каталога этой машине не подходят.")
    if rejected_note:
        out.append(rejected_note)
    out.append("")

    out += ["## Что даст следующее улучшение железа", "", _next_step(choice), ""]
    out += ["## Как откатить", "",
            f"Одна команда: `{ROLLBACK}`. Она останавливает службу и удаляет всё, что установка записала в журнал: настройки, "
            "службу, скачанные сборки; того, чего нет в журнале, она не трогает. Файлы моделей остаются на диске; "
            f"`{ROLLBACK} --models` удалит и их.", ""]
    out += ["## Как сменить выбор модели позже", "",
            f"`{_change_commands(choice)[0]}`  (подставьте другой `--model-id` и `--quant`)", ""]
    if _change_commands(choice)[1]:
        out += [f"Чтобы взять отвергнутую альтернативу: `{_change_commands(choice)[1]}`", ""]
    out += ["Это тот же путь установки: спросит согласие, заново подберёт настройки, сделает проверки и запишет шаги в журнал, "
            "так что откат `shura uninstall` работает и после смены.", ""]
    return "\n".join(out)
