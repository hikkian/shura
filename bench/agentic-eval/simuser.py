"""SIMUSER1: who answers when the agent asks a question (the engine's `question` tool) in the stand.

Today nobody does: the tool call stays pending and the run waits until the task's deadline. This module gives run_one a policy,
chosen with AGENTIC_EVAL_QUESTIONS:
  fixed  — the default, QSTAND1: after 5 s the same text, "the user is not available, pick the most reasonable assumption, write it down
           and go on" (QUESTION_TEXT); QLEV1b and every earlier arm ran with this, so it stays the comparable baseline;
  none   — nobody answers: the question stays pending until the task's deadline (the behaviour before QSTAND1; only logged);
  reject — the request is rejected (the tool returns that the user declined);
  sim    — a model plays the user: Claude through the subscription CLI (`shura-qa claude -p`), AGENTIC_EVAL_SIMUSER_MODEL
           (default claude-haiku-5-5). It knows ONLY the task prompt and the task's `hidden/user_intent.md` (a short statement of
           what the user actually wants) — never the hidden tests, the solution or the project files. It answers like a busy person
           (one or two sentences, or an option label), never solves the task, and says "I don't know, decide yourself" when the
           intent does not cover the question.

Determinism: the CLI has no temperature switch, so sameness between arms comes from a cache keyed by (task id, the questions):
the same question in another arm gets the same answer (AGENTIC_EVAL_SIMUSER_CACHE, default <out>/simuser-cache).
Leak guard: every answer is checked against the task's hidden files (hidden/, solution/): a literal (quoted string of 4+ characters,
number of 3+ digits, identifier of 6+ characters) that occurs there but neither in the prompt, nor in start/, nor in the intent is a
leak — the answer is replaced by "I don't know, decide yourself" and the leak is logged. Every question, answer, source and cost
is written to the agent log as a `question` event and counted in the run record (questions, question_sources, simuser_cost_usd).
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

QUESTION_TEXT = "Пользователь сейчас недоступен. Выбери самое разумное допущение, явно запиши его в ответе и продолжай."   # QSTAND1, the fixed reply
FIXED_WAIT_S = 5                                                                                                          # the user "is busy" for this long
DONT_KNOW = {"ru": "Не знаю, реши сам.", "en": "I don't know, decide yourself."}
CLI_PRE = ["shura-qa", "claude", "-p"]
CLI_POST = ["--output-format", "json", "--no-session-persistence", "--tools", ""]   # the prompt must come before --tools (variadic)

PROMPT = """You are playing the USER who gave a task to a coding agent. The agent paused and asks you something.
You know ONLY two things: the task text you gave, and your private intent (what you actually want). You have not seen the
agent's work and you cannot see any test. Answer like a busy person: one or two short sentences, or just the option label(s).
Rules:
- Never solve the task, never write code or file contents, never dictate steps: only state what you want.
- If an offered option matches your intent, answer with that option's exact label. You may type a short custom answer instead.
- If the question is not covered by the task text or your intent, answer exactly: {dont_know}
- Answer in the language of the task text.
Reply with ONLY a JSON object: {{"answers": [[<answer or label for question 1>], [<for question 2>], ...]}} — one inner list per
question, in order; for a multiple-choice question with `multiple` the inner list may hold several labels.

TASK TEXT (what you asked the agent):
<<<
{prompt}
>>>

YOUR INTENT (private; the agent does not see this):
<<<
{intent}
>>>

THE AGENT ASKS:
{questions}"""


def lang_of(text):
    return "ru" if len(re.findall(r"[А-Яа-яЁё]", text)) > len(re.findall(r"[A-Za-z]", text)) else "en"


def normalize_questions(questions):
    """The cache key material: question texts, headers, option labels, flags — whitespace-insensitive."""
    norm = []
    for q in questions or []:
        norm.append({"q": " ".join(str(q.get("question", "")).split()).lower(), "h": " ".join(str(q.get("header", "")).split()).lower(),
                     "o": [" ".join(str(o.get("label", "")).split()).lower() for o in (q.get("options") or [])],
                     "m": bool(q.get("multiple")), "c": q.get("custom", True) is not False})
    return norm


def cache_key(task_id, questions):
    return hashlib.sha256(json.dumps([task_id, normalize_questions(questions)], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


# ----------------------------------------------------------------------------------------------------------------------- leak guard
_QUOTED = re.compile(r"'((?:[^'\\]|\\.){4,}?)'|\"((?:[^\"\\]|\\.){4,}?)\"")
_NUMBER = re.compile(r"(?<![\w.])\d[\d.,:/-]{2,}(?![\w])")
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
_WORDISH = re.compile(r"[A-Za-zА-Яа-яЁё0-9_.,:/-]+")


def literals(text):
    """The tokens that could carry a hidden answer: quoted strings (4+), numbers (3+ characters), identifiers (6+)."""
    out = set()
    for m in _QUOTED.finditer(text):
        out.add((m.group(1) or m.group(2)).strip().lower())
    for m in _NUMBER.finditer(text):
        out.add(m.group(0).strip(".,:").lower())
    for m in _IDENT.finditer(text):
        out.add(m.group(0).lower())
    return {x for x in out if len(x) >= 3}


def read_tree(root, limit=400_000):
    chunks = []
    if root and Path(root).is_dir():
        for p in sorted(Path(root).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                try:
                    chunks.append(p.read_text(encoding="utf-8", errors="replace")[:limit])
                except OSError:
                    pass
    return "\n".join(chunks)


class LeakGuard:
    """Tokens of the hidden material that the visible material (prompt, start/, intent) does not already contain."""

    def __init__(self, task, intent):
        hidden = read_tree(Path(task["dir"]) / "hidden") + "\n" + read_tree(Path(task["dir"]) / "solution")
        visible = (task.get("prompt") or "") + "\n" + read_tree(Path(task["dir"]) / "start") + "\n" + (intent or "")
        vis = visible.lower()
        self.secret = {t for t in literals(hidden) if t not in vis}

    def leaks(self, answer_text):
        low = answer_text.lower()
        words = set(_WORDISH.findall(low))
        found = set()
        for t in self.secret:
            if t in words or (len(t) >= 6 and t in low):
                found.add(t)
        return sorted(found)


# ----------------------------------------------------------------------------------------------------------------------- the user model
def call_cli(prompt, model, tries=2, timeout=90):
    """One subscription CLI call; returns (text, usage dict). Raises RuntimeError when every try fails."""
    err = ""
    for _ in range(tries):
        try:
            r = subprocess.run(CLI_PRE + [prompt, "--model", model] + CLI_POST, capture_output=True, text=True, timeout=timeout,
                               cwd="/dev/shm", stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            err = str(e)
            continue
        try:
            obj = json.loads(r.stdout)
            usage = {"cost_usd": obj.get("total_cost_usd"), "turns": obj.get("num_turns"), "usage": obj.get("usage")}
            text = obj.get("result") or ""
        except ValueError:
            usage, text = {}, r.stdout
        if text.strip():
            return text, usage
        err = (r.stderr or r.stdout)[-300:]
    raise RuntimeError(f"claude CLI gave no answer: {err}")


def parse_answers(text, n):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    ans = obj.get("answers")
    if not isinstance(ans, list) or len(ans) != n:
        return None
    out = []
    for a in ans:
        if isinstance(a, str):
            a = [a]
        if not isinstance(a, list) or not all(isinstance(x, str) for x in a):
            return None
        out.append([x.strip() for x in a if x.strip()] or [""])
    return out


def render_questions(questions):
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. [{q.get('header', '')}] {q.get('question', '')}")
        for o in q.get("options") or []:
            lines.append(f"   - option \"{o.get('label', '')}\": {o.get('description', '')}")
        if q.get("multiple"):
            lines.append("   (several labels may be chosen)")
    return "\n".join(lines)


class QuestionPolicy:
    """Answers the engine's question requests for one run; collects the Q&A for the agent log and the run record."""

    def __init__(self, task, mode=None, out_dir=None, model=None, cache_dir=None):
        self.task = task
        self.mode = (mode or os.environ.get("AGENTIC_EVAL_QUESTIONS", "fixed")).strip().lower()
        if self.mode not in ("fixed", "none", "reject", "sim"):
            raise ValueError(f"AGENTIC_EVAL_QUESTIONS must be fixed, none, reject or sim, not {self.mode!r}")
        self.wait_s = FIXED_WAIT_S if self.mode == "fixed" else 0      # fixed: the user is busy for a moment; sim answers at once
        self.model = model or os.environ.get("AGENTIC_EVAL_SIMUSER_MODEL", "claude-haiku-5-5")
        cache = cache_dir or os.environ.get("AGENTIC_EVAL_SIMUSER_CACHE") or (str(Path(out_dir) / "simuser-cache") if out_dir else None)
        self.cache_dir = Path(cache) if cache else None
        intent_path = Path(task["dir"]) / "hidden" / "user_intent.md"
        self.intent = intent_path.read_text(encoding="utf-8").strip() if intent_path.is_file() else ""
        self.lang = lang_of(task.get("prompt") or "")
        self.guard = LeakGuard(task, self.intent) if self.mode == "sim" else None
        self.events = []
        self.cost = 0.0

    # -- the one entry point the harness calls
    def answer(self, request):
        """request: the engine's QuestionRequest. Returns ("reply", answers) | ("reject", None) | ("none", None)."""
        questions = request.get("questions") or []
        n = max(1, len(questions))
        t0 = time.monotonic()
        source, answers, leaks, usage = self.mode, None, [], {}
        if self.mode == "fixed":
            answers = [[QUESTION_TEXT] for _ in range(n)]
        elif self.mode == "reject":
            pass
        elif self.mode == "sim":
            answers, source, leaks, usage = self._sim(questions)
        event = {"type": "question", "request_id": request.get("id"), "questions": questions, "answers": answers, "source": source,
                 "leaks": leaks, "cost_usd": usage.get("cost_usd"), "seconds": round(time.monotonic() - t0, 2)}
        self.events.append(event)
        if usage.get("cost_usd"):
            self.cost += float(usage["cost_usd"])
        if self.mode == "reject":
            return "reject", None
        if answers is None:
            return "none", None
        return "reply", answers

    def _sim(self, questions):
        key = cache_key(self.task["id"], questions)
        if self.cache_dir:
            hit = self.cache_dir / f"{key}.json"
            if hit.is_file():
                c = json.loads(hit.read_text(encoding="utf-8"))
                return c["answers"], "cache", c.get("leaks", []), {}
        prompt = PROMPT.format(dont_know=json.dumps(DONT_KNOW[self.lang], ensure_ascii=False), prompt=(self.task.get("prompt") or "")[:12000],
                               intent=self.intent or "(nothing beyond the task text — if the task text does not say, you do not know)",
                               questions=render_questions(questions))
        try:
            text, usage = call_cli(prompt, self.model)
        except RuntimeError as e:
            answers = [[DONT_KNOW[self.lang]] for _ in questions]
            return answers, f"model-failed: {str(e)[:120]}", [], {}
        answers = parse_answers(text, len(questions))
        if answers is None:                       # not the agreed shape: don't know, and not cached (a later arm may get a proper answer)
            return [[DONT_KNOW[self.lang]] for _ in questions], "model-unparsable", [], usage
        leaks = self.guard.leaks(" ".join(" ".join(a) for a in answers)) if self.guard else []
        source = "model"
        if leaks:
            answers = [[DONT_KNOW[self.lang]] for _ in questions]
            source = "leak-guard"
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"task": self.task["id"], "questions": questions, "answers": answers,
                                                                   "source": source, "leaks": leaks, "model": self.model}, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        return answers, source, leaks, usage

    def stats(self):
        return {"questions": len(self.events), "question_sources": sorted({e["source"].split(":")[0] for e in self.events}),
                "question_leaks": sum(1 for e in self.events if e["leaks"]), "simuser_cost_usd": round(self.cost, 5), "question_policy": self.mode}


def questions_summary(results):
    """Over a results.jsonl: how often the agent asked, and resolved rates with and without a question (per arm)."""
    out = {}
    for r in results:
        a = out.setdefault(r.get("arm", "?"), {"runs": 0, "asked_runs": 0, "questions": 0, "leaks": 0, "resolved_asked": 0, "resolved_silent": 0, "cost_usd": 0.0})
        a["runs"] += 1
        q = int(r.get("questions") or 0)
        a["questions"] += q
        a["leaks"] += int(r.get("question_leaks") or 0)
        a["cost_usd"] += float(r.get("simuser_cost_usd") or 0.0)
        if q:
            a["asked_runs"] += 1
            a["resolved_asked"] += int(bool(r.get("resolved")))
        else:
            a["resolved_silent"] += int(bool(r.get("resolved")))
    for a in out.values():
        silent = a["runs"] - a["asked_runs"]
        a["resolved_rate_asked"] = round(a["resolved_asked"] / a["asked_runs"], 3) if a["asked_runs"] else None
        a["resolved_rate_silent"] = round(a["resolved_silent"] / silent, 3) if silent else None
        a["cost_usd"] = round(a["cost_usd"], 4)
    return out


def load_results(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def usefulness(runs_by_label, needs_question):
    """The asked-or-not comparison per (label, arm) over tasks that need a question and the rest: runs, share that asked, mean score and
    resolved rate, questions per run, leaks, simulator cost. `runs_by_label` = {label: [result rows]}; `needs_question` = set of task ids."""
    table = {}
    for label, rows in runs_by_label.items():
        for r in rows:
            for group in ("need" if r.get("task") in needs_question else "other",):
                c = table.setdefault((label, r.get("arm", "?"), group), {"runs": 0, "asked": 0, "questions": 0, "leaks": 0, "resolved": 0, "score": 0.0,
                                                                         "cost_usd": 0.0, "timed_out": 0})
                c["runs"] += 1
                q = int(r.get("questions") or 0)
                c["asked"] += 1 if q else 0
                c["questions"] += q
                c["leaks"] += int(r.get("question_leaks") or 0)
                c["resolved"] += int(bool(r.get("resolved")))
                c["score"] += float(r.get("score") or 0.0)
                c["cost_usd"] += float(r.get("simuser_cost_usd") or 0.0)
                c["timed_out"] += int(bool(r.get("timed_out")))
    return table


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="SIMUSER1: did asking help? Compares result files of runs made with different AGENTIC_EVAL_QUESTIONS.")
    ap.add_argument("results", nargs="+", help="LABEL=results.jsonl (e.g. fixed=out-fixed/results.jsonl sim=out-sim/results.jsonl)")
    ap.add_argument("--tasks-root", default=str(Path(__file__).resolve().parent / "tasks_general"))
    a = ap.parse_args(argv)
    import harness
    need = {t["id"] for t in harness.all_tasks(Path(a.tasks_root)) if t.get("needs_question")}
    table = usefulness({x.split("=", 1)[0]: load_results(x.split("=", 1)[1]) for x in a.results}, need)
    print(f"{'policy':8} {'arm':14} {'tasks':6} {'runs':>4} {'asked':>5} {'q/run':>6} {'resolved':>8} {'score':>6} {'timeouts':>8} {'leaks':>5} {'cost$':>7}")
    for (label, arm, group), c in sorted(table.items()):
        n = c["runs"]
        print(f"{label:8} {arm:14} {group:6} {n:4d} {c['asked']:5d} {c['questions'] / n:6.2f} {c['resolved'] / n:8.2f} {c['score'] / n:6.2f} "
              f"{c['timed_out']:8d} {c['leaks']:5d} {c['cost_usd']:7.3f}")


if __name__ == "__main__":
    main()
