"""User switches that survive a gateway restart: the context mode (SWITCHUI1) and the thinking level (THINK2).

Each one is a small flag file in the state directory, written only when the user changes it (no repeated disk writes).
Absent flag = the default, which keeps the old behaviour byte for byte."""
import json
from pathlib import Path

CONTEXT_MODES = ("auto", "extended")   # auto: short layout, the long one when the request outgrows it; extended: long from the start
THINK_LEVELS = ("off", "low", "medium", "high", "max")   # EFFORT1: the slider the user moves; max added
CLIENT_THINK_KEYS = ("enable_thinking", "reasoning_effort")
# EFFORT1: an OpenAI reasoning_effort in the client's request names the level for that one request
EFFORT_ALIASES = {"none": "off", "off": "off", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
                  "xhigh": "max", "max": "max"}
BUDGET_KEYS = {"targetTokens", "ceilingTokens", "message"}
CLIENT_BUDGET_KEYS = ("reasoning_budget_tokens", "thinking_budget_tokens")
DEFAULT_WRAP_UP = "Thinking time is up. I will stop here and answer with my best current understanding."


def read_flag(path, allowed, log):
    try:
        value = Path(path).read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        log(f"State flag ignored: {Path(path).name} ({type(error).__name__}: {error})")
        return None
    if value not in allowed:
        log(f"State flag ignored: {Path(path).name} (unknown value {value!r})")
        return None
    return value


def write_flag(path, value):
    """Persist a switch; None removes the flag (back to the default)."""
    path = Path(path)
    if value is None:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def validate_think_config(models):
    """thinkLevels: level -> chat_template_kwargs the level adds; thinkFallback: a level the template lacks -> the one used;
    effortBudgets (EFFORT1): level -> {targetTokens, ceilingTokens, message}, the reasoning budget the level sends to the server."""
    for model_id, settings in models.items():
        validate_effort_budgets(model_id, settings.get("effortBudgets"), settings.get("outputLimitTokens"),
                                 settings.get("effortBudgetMarginTokens", 0))
        levels = settings.get("thinkLevels")
        fallback = settings.get("thinkFallback")
        if levels is None:
            if fallback is not None:
                raise ValueError(f"Invalid model config for {model_id}: thinkFallback needs thinkLevels")
            continue
        if not isinstance(levels, dict) or not set(levels) <= set(THINK_LEVELS):
            raise ValueError(f"Invalid model config for {model_id}: thinkLevels may only set {list(THINK_LEVELS)}")
        for level, kwargs in levels.items():
            if not isinstance(kwargs, dict) or not all(
                    isinstance(key, str) and isinstance(value, (str, bool, int, float)) for key, value in kwargs.items()):
                raise ValueError(f"Invalid model config for {model_id}: thinkLevels.{level} must map names to plain values")
        if fallback is None:
            continue
        if (not isinstance(fallback, dict) or not set(fallback) <= set(THINK_LEVELS)
                or any(target not in levels for target in fallback.values())):
            raise ValueError(f"Invalid model config for {model_id}: thinkFallback must map levels to levels set in thinkLevels")


def validate_effort_budgets(model_id, budgets, output_limit=None, margin=0):
    if output_limit is not None and (type(output_limit) is not int or output_limit < 0):
        raise ValueError(f"Invalid model config for {model_id}: outputLimitTokens must be a non-negative integer or null")
    if type(margin) is not int or margin < 0:
        raise ValueError(f"Invalid model config for {model_id}: effortBudgetMarginTokens must be a non-negative integer")
    if budgets is None:
        return
    if not isinstance(budgets, dict) or not set(budgets) <= set(THINK_LEVELS):
        raise ValueError(f"Invalid model config for {model_id}: effortBudgets may only set {list(THINK_LEVELS)}")
    for level, budget in budgets.items():
        if not isinstance(budget, dict) or not set(budget) <= BUDGET_KEYS:
            raise ValueError(f"Invalid model config for {model_id}: effortBudgets.{level} may only set {sorted(BUDGET_KEYS)}")
        target, ceiling = budget.get("targetTokens"), budget.get("ceilingTokens")
        for name, value in (("targetTokens", target), ("ceilingTokens", ceiling)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid model config for {model_id}: effortBudgets.{level}.{name} must be a non-negative integer or null")
        if target is not None and ceiling is not None and target > ceiling:
            raise ValueError(f"Invalid model config for {model_id}: effortBudgets.{level}: targetTokens must not exceed ceilingTokens")
        if ceiling is not None and output_limit is not None and ceiling + margin > output_limit:
            raise ValueError(f"Invalid model config for {model_id}: effortBudgets.{level}.ceilingTokens + margin ({margin}) exceeds output limit ({output_limit})")
        if "message" in budget and not isinstance(budget["message"], str):
            raise ValueError(f"Invalid model config for {model_id}: effortBudgets.{level}.message must be a string")


def effective_think_level(model_config, level):
    """The level the model really runs at, or None when the model has no thinking levels (requests stay as they are)."""
    levels = model_config.get("thinkLevels")
    if not levels:
        return None
    level = level or "medium"
    if level in levels:
        return level
    return (model_config.get("thinkFallback") or {}).get(level, "medium")


def budget_for(model_config, level):
    """The reasoning budget of a level (EFFORT1): {"target", "ceiling", "message"} from effortBudgets, or None when the model has
    none for it. A level the template lacks keeps the budget of the level it falls back to."""
    budgets = model_config.get("effortBudgets") or {}
    chosen = effective_think_level(model_config, level) if model_config.get("thinkLevels") else (level or "medium")
    budget = budgets.get(chosen) if chosen else None
    if budget is None:
        return None
    return {"level": chosen, "target": budget.get("targetTokens"), "ceiling": budget.get("ceilingTokens"),
            "message": budget.get("message", DEFAULT_WRAP_UP)}


def client_effort_level(obj):
    """The level a client asked for with the OpenAI field reasoning_effort, or None (unknown values are left to the server)."""
    value = obj.get("reasoning_effort")
    return EFFORT_ALIASES.get(value.lower()) if isinstance(value, str) else None


def apply_think(body, level, model_config, budgets=False):
    """Add the level's chat_template_kwargs to a chat request. The default level adds nothing, and a request that already
    names a thinking setting keeps it (the client wins).

    EFFORT1, with budgets=True (guardian effortBudgets): the level's reasoning budget goes with the request as
    reasoning_budget_tokens (the ceiling), reasoning_budget_message (the wrap-up line the server forces before </think>)
    and reasoning_budget_soft_tokens (the target below which the server never intervenes). A client's reasoning_effort names
    the level for that request; a client's own budget fields are kept."""
    if level is None and not budgets:
        return body
    try:
        obj = json.loads(body)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return body
    if not isinstance(obj, dict):
        return body
    existing = obj.get("chat_template_kwargs")
    existing = existing if isinstance(existing, dict) else {}
    client_level = client_effort_level(obj)
    client_sets_thinking = "reasoning_effort" in obj or any(key in existing for key in CLIENT_THINK_KEYS)
    changed = False
    if level is not None and not client_sets_thinking:
        effective = effective_think_level(model_config, level)
        kwargs = (model_config.get("thinkLevels") or {}).get(effective) if effective else None
        if kwargs:
            obj["chat_template_kwargs"] = {**existing, **kwargs}
            changed = True
    if budgets and not any(key in obj for key in CLIENT_BUDGET_KEYS):
        budget = budget_for(model_config, client_level or level)
        if budget is not None and budget["level"] != "off" and budget["ceiling"] is not None:
            obj["reasoning_budget_tokens"] = budget["ceiling"]
            obj["reasoning_budget_message"] = budget["message"]
            if budget["target"] is not None:
                obj["reasoning_budget_soft_tokens"] = budget["target"]
            changed = True
    return json.dumps(obj).encode() if changed else body


def think_snapshot(model_id, model_config, level, budgets=False):
    levels = model_config.get("thinkLevels") or {}
    budget = budget_for(model_config, level)
    return {"level": level or "medium", "default": level is None, "model": model_id,
            "effective": effective_think_level(model_config, level),
            "available": [name for name in THINK_LEVELS if name in levels],
            "fallback": dict(model_config.get("thinkFallback") or {}),
            "levels": list(THINK_LEVELS), "budgets": bool(budgets),
            "budget": ({"target": budget["target"], "ceiling": budget["ceiling"]} if budgets and budget else None)}


def context_snapshot(mode, model_id, model_config, layout):
    short = model_config.get("ctxSize")
    long_ctx = (model_config.get("longContextProfile") or {}).get("ctxSize")
    return {"mode": mode, "model": model_id, "available": bool(long_ctx),
            "window": long_ctx if layout == "long" and long_ctx else short,
            "short_ctx": short, "long_ctx": long_ctx,
            "switch_tokens": model_config.get("longContextSwitchTokens", 0)}
