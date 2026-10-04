"""Add a temporary, CPU-only OpenAI-compat model handler to a pinned bfcl-eval venv.

The upstream BFCL CLI eagerly imports every hosted/local model handler. On a host
without a GPU this pulls optional audio/transformers/CUDA packages just to register
one OpenAI-compatible server. This script changes only the caller's disposable venv.
"""
from importlib.metadata import version
from pathlib import Path
import argparse

PIN = "2025.12.17"
HANDLER = '''import os
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

class ShuraOpenAIHandler(OpenAICompletionsHandler):
    """BFCL FC handler for llama-server /v1/chat/completions."""
    def _query_FC(self, inference_data):
        kwargs = {
            "messages": inference_data["message"],
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": float(os.environ.get("SHURA_TOP_P", "0.95")),
            "max_tokens": int(os.environ.get("SHURA_MAX_TOKENS", "32768")),
            "extra_body": {"top_k": int(os.environ.get("SHURA_TOP_K", "20"))},
            "store": False,
        }
        penalty = os.environ.get("SHURA_PRESENCE_PENALTY")
        if penalty is not None:
            kwargs["presence_penalty"] = float(penalty)
        if inference_data.get("tools"):
            kwargs["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "tools": inference_data.get("tools", []),
        }
        return self.generate_with_backoff(**kwargs)
'''
MODEL_CONFIG = '''from dataclasses import dataclass
from typing import Optional
from bfcl_eval.model_handler.api_inference.shura_openai import ShuraOpenAIHandler

@dataclass
class ModelConfig:
    model_name: str
    display_name: str
    url: str
    org: str
    license: str
    model_handler: object
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    is_fc_model: bool = True
    underscore_to_dot: bool = False

MODEL_CONFIG_MAPPING = {
    name: ModelConfig(
        model_name=name,
        display_name=name,
        url="http://127.0.0.1",
        org="Shura",
        license="local",
        model_handler=ShuraOpenAIHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    for name in ("shura-tiel", "shura-occamy")
}
'''


def prepare(site_packages: Path, allow_version: str = PIN) -> Path:
    if version("bfcl-eval") != allow_version:
        raise RuntimeError(f"expected bfcl-eval {allow_version}, found {version('bfcl-eval')}")
    root = site_packages / "bfcl_eval"
    model_dir = root / "model_handler" / "api_inference"
    model_dir.joinpath("shura_openai.py").write_text(HANDLER, encoding="utf-8")
    (root / "constants" / "model_config.py").write_text(MODEL_CONFIG, encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--version", default=PIN)
    args = parser.parse_args()
    print(prepare(args.site_packages, args.version))


if __name__ == "__main__":
    main()
