"""RSIL0b --fake: a stand-in for the `claude -p` binary (and, RSI1, for `codex exec`: called as `stub exec ... -o FILE -`, it writes the
reply to FILE and the token total to stderr the way codex does). It calls no model and costs nothing (the usage it reports is fixed). It answers by the
role line of the prompt ('# role: <name>') and writes its proposals as real unified diffs against RSI_FAKE_BASE (the harness.md the pilot
is editing). The marker it proposes is the one fake_worker.py looks for. RSI_FAKE_COMPOSER=keep makes the composer keep the base.
"""
import difflib
import json
import os
import sys
from pathlib import Path

INV = "- FAKE-MARK-7: finish only after the project's tests pass.\n"
DOM = "- FAKE-DOM: a practitioner note for this family.\n"


def diff_against_base(addition):
    base_path = os.environ.get("RSI_FAKE_BASE", "")
    base = Path(base_path).read_text(encoding="utf-8") if base_path and os.path.exists(base_path) else ""
    new = base + addition
    d = "".join(difflib.unified_diff(base.splitlines(keepends=True), new.splitlines(keepends=True), "a/harness.md", "b/harness.md"))
    return "```diff\n" + d + "```"


def reply(prompt):
    role = next((line.split(":", 1)[1].strip() for line in prompt.splitlines() if line.startswith("# role:")), "")
    if role == "invariant":
        return diff_against_base(INV) + "\nINV: after the last edit, run the project's tests and read the output (stub note).\n"
    if role == "cause":            # RSI2D: the cause editor of cycle2 proposes the same marker (fake_worker scores by it)
        return diff_against_base(INV) + "\nCAUSE: the stub's one edit for this cause (stub note).\n"
    if role == "specific":
        return diff_against_base(DOM) + "\nDOM: a practitioner fact (stub note).\n"
    if role == "composer":
        if os.environ.get("RSI_FAKE_COMPOSER") == "keep":
            return "KEEP_BASE\ndropped -> invariant -> stub: the composer keeps the base in this test\n"
        return diff_against_base(INV) + "\nkept -> invariant -> stub: the invariant candidate, as the composer's whole answer\n"
    if role == "reflector":
        return "Stub experience entry.\n- what changed: the candidate of this round (stub)\n- the next question: does the marker move the score?\n"
    return "unknown role in the stub"


def main():
    prompt = sys.stdin.read()            # the editors send the prompt on stdin (RSIR1 D1), as the real binary is called
    if sys.argv[1:2] == ["exec"]:
        out_file = sys.argv[sys.argv.index("-o") + 1]
        Path(out_file).write_text(reply(prompt), encoding="utf-8")
        print("tokens used\n1,200", file=sys.stderr)
        return
    out = {"result": reply(prompt), "is_error": False, "total_cost_usd": 0.0,
           "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
