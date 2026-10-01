"""Downloads the candidate models for the comparison (resumable, size and SHA-256 verified, low priority).
    python3 bench/agentic-eval/fetch_models.py [--dest DIR]
The files are the IQ4_XS builds (the size class we run in real life): the base model (unsloth, with the MTP head) and Occamy."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "installer"))
from universal import modelstore  # noqa: E402

CANDIDATES = {
    "base": ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"),
    "occamy": ("Accio-Lab/occamy-1.0-GGUF", "occamy-1.0-IQ4_XS.gguf"),
}


def fetch(dest, names=("base", "occamy")):
    dest = Path(dest)
    paths = {}
    for name in names:
        repo, fname = CANDIDATES[name]
        size, sha = modelstore.listing(repo)[fname]
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        print(f"{name}: {fname} ({size / 1e9:.1f} GB)", flush=True)
        paths[name] = str(modelstore.fetch(url, dest / fname, size, sha,
                                           progress=lambda d, t: print(f"  {d / 1e9:.1f}/{t / 1e9:.1f} GB", flush=True)))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path.home() / "Desktop/AI/models/eval"))
    print(fetch(ap.parse_args().dest))
