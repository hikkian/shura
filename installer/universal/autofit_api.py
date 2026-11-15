"""One interface of the installer's autofit (INSTALL1c, step 3).

    choose(hw, catalog, engine_caps, *, profile=None) -> Choice

Input: the machine description (the profile of detect.py, checked by schema.validate_hardware), the catalog
(catalog/models.json: only `models` are choosable, `reference` never is) and the engine capabilities (the backends the engine
can run). Output: a Choice with the mode (model, quant, backend, context, kind of mode), the speed forecast (a range of decode
tok/s, or "not measured"), the reasons (what limited the choice: video memory, RAM, disk, speed) and the rejected alternative
with numbers. Behind the interface the present planner (planner.py) does the sums; nothing of it is rewritten here.

Contract: choose() never raises on a machine without a GPU (CPU path) and never raises when nothing fits (it returns a Choice
with ok=False and the reason); the same input gives the same output; `reference` entries are never chosen.
"""
import copy
from dataclasses import dataclass, field, asdict

from . import planner
from .schema import validate_hardware, validate_catalog

GiB = 1024 ** 3
NOT_MEASURED = "не измерено"
DISK_MARGIN = 1 * GiB           # kept free on top of the model file
BACKENDS = ("hip", "cuda", "vulkan", "metal", "cpu")
_TO_PLANNER = {"hip": "rocm"}   # our name -> the planner's name (the planner says rocm for HIP)


@dataclass
class Choice:
    ok: bool
    model_id: str = None
    quant: str = None
    backend: str = None          # hip | cuda | vulkan | metal | cpu
    context: int = 0             # tokens of window
    mode: str = "none"           # kind of mode from the planner: gpu | hybrid | cpu | unified (| "none" when nothing fits)
    speed_range: dict = field(default_factory=lambda: {"status": NOT_MEASURED, "measured": False})
    reasons: list = field(default_factory=list)   # what limited the choice, in words with numbers
    alternative: dict = None     # the runner-up with numbers and why it was not taken, or None (then `notes` say why)
    notes: list = field(default_factory=list)
    limiting: str = None         # vram | ram | disk | speed | None
    backend_order: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _caps(engine_caps):
    """Normalised set of backends the engine can run; None = no restriction."""
    if engine_caps is None:
        return None
    if isinstance(engine_caps, dict):
        engine_caps = engine_caps.get("backends", [])
    return {("hip" if b == "rocm" else b) for b in engine_caps}


def _rocm_installed(hw):
    drivers = (hw.get("system") or {}).get("drivers") or {}
    if drivers.get("rocm"):
        return True
    if "system" in hw:      # extended detection ran and found no ROCm: do not trust the card's static list
        return False
    return any("rocm" in (g.get("backends") or []) for g in hw.get("gpus", []) if g.get("vendor") == "amd")


def backend_order(hw, engine_caps=None):
    """Backends to try, in the orchestrator's order: AMD with ROCm installed -> hip first, then vulkan; AMD without ROCm and
    Intel -> vulkan; NVIDIA -> cuda, then vulkan; Apple -> metal; cpu always last. Only backends the engine can run are kept
    (engine_caps None = all). A machine without a GPU gets ["cpu"]."""
    caps = _caps(engine_caps)
    gpus = hw.get("gpus") or []
    order = []
    if gpus:
        vendor = max(gpus, key=lambda g: g.get("vram_total", 0)).get("vendor")
        if vendor == "nvidia":
            order = ["cuda", "vulkan"]
        elif vendor == "amd":
            order = (["hip"] if _rocm_installed(hw) else []) + ["vulkan"]
        elif vendor == "apple":
            order = ["metal"]
        else:
            order = ["vulkan"]
    order = [b for b in order if caps is None or b in caps]
    return order + ["cpu"]      # the CPU is the floor: it is kept even when the caller's list forgets it


def _clean_catalog(hw, catalog):
    """Choosable models only: `reference` is dropped (and any model sharing an id with a reference entry); quants that do not
    fit the free disk are dropped, and a model whose lowest allowed quant does not fit the disk is dropped with a reason."""
    ref_ids = {r.get("id") for r in catalog.get("reference", []) or []}
    free = (hw.get("disk") or {}).get("free")
    models, disk_notes = [], []
    for m in catalog.get("models", []):
        if m.get("id") in ref_ids:
            continue
        m = copy.deepcopy(m)
        if free is not None:
            fits = [q for q in m["quants"] if q["file_bytes"] + DISK_MARGIN <= free]
            floor = next((q for q in m["quants"] if q["id"] == m.get("min_quant")), min(m["quants"], key=lambda q: q["quality"]))
            if floor["file_bytes"] + DISK_MARGIN > free:
                disk_notes.append(f"{m['name']}: диск: свободно {free / GiB:.1f} ГиБ, нужен минимум "
                                  f"{(floor['file_bytes'] + DISK_MARGIN) / GiB:.1f} ГиБ (квант {floor['id']})")
                continue
            m["quants"] = fits
        models.append(m)
    return {**{k: v for k, v in catalog.items() if k not in ("models", "reference")}, "models": models}, disk_notes


def _run_planner(hw, cat, backend, profile, calibration):
    cfg = {}
    if backend == "cpu":
        hw = {**hw, "gpus": []}
    elif backend != "vulkan":
        cfg = {"backend": _TO_PLANNER.get(backend, backend), "native_backends": True}
    if calibration and calibration.get("speed_scale"):
        cfg["speed_scale"] = calibration["speed_scale"]
    return planner.plan(hw, cat, config=cfg, profile=profile or "balanced", menu=False)


def _speed(p, calibration):
    t = p.get("predicted_tok_s")
    if not t:
        return {"status": NOT_MEASURED, "measured": False}
    measured = bool(calibration and calibration.get("measured"))
    return {"low": t["low"], "mid": t["mid"], "high": t["high"], "unit": "tok/s", "measured": measured,
            "status": "измерено на этой машине" if measured else "прогноз, не измерено на этой машине",
            "confidence": p.get("confidence")}


def _limits(hw, p, cat):
    """(limiting factor, reasons). The factor with the highest use of its budget; speed when the target was not reached."""
    mem, reasons = p["memory"], []
    shares = {}
    if mem["vram_budget"] and p["mode"] != "cpu":
        shares["vram"] = mem["vram_need"] / mem["vram_budget"]
        reasons.append(f"видеопамять: нужно {mem['vram_need'] / GiB:.1f} из {mem['vram_budget'] / GiB:.1f} ГиБ доступных")
    if mem["ram_budget"]:
        shares["ram"] = mem["ram_need"] / mem["ram_budget"]
        reasons.append(f"ОЗУ: нужно {mem['ram_need'] / GiB:.1f} из {mem['ram_budget'] / GiB:.1f} ГиБ доступных")
    m = next(x for x in cat["models"] if x["id"] == p["model"])
    q = next(x for x in m["quants"] if x["id"] == p["quant"])
    free = (hw.get("disk") or {}).get("free")
    if free:
        shares["disk"] = (q["file_bytes"] + DISK_MARGIN) / free
        reasons.append(f"диск: файл {q['file_bytes'] / GiB:.1f} ГиБ, свободно {free / GiB:.1f} ГиБ")
    limiting = max(shares, key=lambda k: (shares[k], k)) if shares else None
    if p.get("speed_rung") not in ("target", None):
        reasons.append(f"скорость: цель недостижима на этой машине, прогноз {p['predicted_tok_s']['mid']} tok/s ({p['speed_rung']})")
        limiting = "speed"
    reasons.insert(0, f"ограничивает: {limiting}" if limiting else "ограничение не определено")
    return limiting, reasons + list(p.get("reasons", []))


def _alt_row(p, backend):
    return {"model_id": p["model"], "quant": p["quant"], "backend": backend, "context": p["settings"]["context"],
            "mode": p["mode"], "tok_s": p["predicted_tok_s"]["mid"]}


def _alternative(order, plans, chosen_backend, p, notes):
    """The runner-up: the next backend in the orchestrator's order that also gives a plan; else the planner's best other
    quant/model; else None (and the notes say why)."""
    for b in order:
        if b == chosen_backend:
            continue
        q = plans.get(b)
        if q and q["ok"]:
            row = _alt_row(q, b)
            a, c = row["tok_s"], p["predicted_tok_s"]["mid"]
            if a > c:
                row["why_not"] = (f"быстрее ({a} против {c} tok/s), но {b} стоит ниже {chosen_backend} в порядке бэкендов "
                                  f"оркестратора; скорость прогнозная, не измерена")
            else:
                row["why_not"] = f"не быстрее ({a} против {c} tok/s) и {b} стоит ниже {chosen_backend} в порядке бэкендов"
            return row
        if q:
            notes.append(f"{b}: не подошёл ({q['reasons'][0]})")
    for a in p.get("alternatives", []):
        row = {"model_id": a["model"], "quant": a["quant"], "backend": chosen_backend, "context": a["context"],
               "tok_s": a["tok_s"]}
        row["why_not"] = (f"счёт планировщика ниже: {a['tok_s']} против {p['predicted_tok_s']['mid']} tok/s, "
                          f"окно {a['context']} против {p['settings']['context']}")
        return row
    notes.append("альтернативы нет: другой бэкенд, квант или модель в каталоге не подходят этой машине")
    return None


def _none(reasons, notes, order):
    return Choice(ok=False, reasons=list(reasons), notes=list(notes), backend_order=list(order))


def choose(hw, catalog, engine_caps=None, *, profile=None, calibration=None):
    """The autofit choice for this machine. See the module doc for the contract. `profile` is a planner profile name
    (balanced, fast, long, quality); `calibration` is the dict calibrate() returns (a measured speed_scale corrects the forecast)."""
    try:
        problems = validate_hardware(hw) + validate_catalog(catalog if isinstance(catalog, dict) else {})
    except Exception as e:      # a malformed input is a reason, not a crash
        problems = [f"не удалось проверить вход: {e}"]
    if problems:
        return _none(["входные данные неверны: " + "; ".join(problems[:5])], [], [])
    order = backend_order(hw, engine_caps)
    cat, disk_notes = _clean_catalog(hw, catalog)
    notes = list(disk_notes)
    if not cat["models"]:
        return _none(["ни одна модель каталога не подходит: " + ("; ".join(disk_notes) or "в каталоге нет выбираемых моделей")],
                     notes, order)
    plans = {}
    for b in order:
        try:
            plans[b] = _run_planner(hw, cat, b, profile, calibration)
        except ValueError as e:
            return _none([f"планировщик отклонил вход: {e}"], notes, order)
    first = next((b for b in order if plans[b]["ok"]), None)
    if first is None:
        why = "; ".join(f"{b}: {plans[b]['reasons'][0]}" for b in order)
        return _none([f"ни одна модель не подходит этой машине ({why})"] + disk_notes, notes, order)
    p = plans[first]
    limiting, reasons = _limits(hw, p, cat)
    for w in p.get("warnings", []):
        notes.append(w)
    alt = _alternative(order, plans, first, p, notes)
    return Choice(ok=True, model_id=p["model"], quant=p["quant"], backend=first, context=p["settings"]["context"],
                  mode=p["mode"], speed_range=_speed(p, calibration), reasons=reasons, alternative=alt, notes=notes,
                  limiting=limiting, backend_order=order)


def calibrate(hw=None, choice=None, *, runner=None):
    """Calibration: a short measurement on the machine (measured tok/s / predicted tok/s -> speed_scale). A stub for now: it
    measures nothing and says so. `runner`, when given, is called as runner(hw, choice) and must return a measured tok/s."""
    if runner is not None and choice is not None and choice.ok and choice.speed_range.get("mid"):
        try:
            tok = float(runner(hw, choice))
            return {"measured": True, "status": "измерено", "tok_s": tok,
                    "speed_scale": round(tok / choice.speed_range["mid"], 3), "reason": None}
        except Exception as e:
            return {"measured": False, "status": NOT_MEASURED, "tok_s": None, "speed_scale": None,
                    "reason": f"замер не удался: {e}"}
    return {"measured": False, "status": NOT_MEASURED, "tok_s": None, "speed_scale": None,
            "reason": "замер на машине ещё не реализован: прогноз остаётся прогнозом планировщика"}


def recalibrate(hw, catalog, engine_caps=None, *, profile=None, calibration=None):
    """Repeat the choice with a new profile (and the calibration, when there is a measured one). Same contract as choose()."""
    return choose(hw, catalog, engine_caps, profile=profile, calibration=calibration or calibrate(hw))
