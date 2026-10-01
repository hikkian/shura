"""Getting a llama.cpp build onto the machine and proving that it works: download, verify, extract, self-test.

Safety rules: HTTPS only; every download is checked against a SHA-256 (the release digest, or a pinned value for
the tiny test model); archives are extracted with path-traversal checks; the self-test runs the server on
127.0.0.1 with a tiny model and always kills it. Nothing here needs root or writes outside the directory you pass in.
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from . import backends

PINNED_TAG = "b11301"            # known-good llama.cpp release (2026-09-30); `--latest` asks GitHub instead
RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
# The TurboQuant+ fork of llama.cpp (turbo2/3/4 KV cache, adaptive expert cache, MTP) on Vulkan, Metal and CUDA. Third-party
# code: the release is pinned and every archive must match the SHA-256 recorded here (taken from the release on 2026-10-01),
# whatever the release page says later. `shura install --no-turbo` never uses it.
TQP_REPO = "TheTom/llama-cpp-turboquant"
TQP_TAG = "tqp-v0.4.0"
TQP_SHA256 = {
    "turboquant-plus-tqp-v0.4.0-linux-x64-cpu.tar.gz": "f9f55bcc7b7baa5eb7f0e8768d96e9e46bdb7df5705418d50ab2f6cb737a5920",
    "turboquant-plus-tqp-v0.4.0-linux-x64-vulkan.tar.gz": "6b9c8929c9f509b843c401e8eb532ed8a28031bd6acf6c925458011db5d9d9d5",
    "turboquant-plus-tqp-v0.4.0-macos-arm64-metal.tar.gz": "f38d0c68e2db29b705ec3944360296c1f8003e4f920d60e6740f9968f3f93769",
    "turboquant-plus-tqp-v0.4.0-windows-x64-cuda12.4.zip": "033c0a2ce4d876bd1f4c840f1c6eef9a3a55f6af54e20f61bde59e7b2181e6d7",
}
SMOKE_MODEL = {                  # 19 MB story model used only to prove a build runs; never recommended to users
    "url": "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
    "sha256": "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
    "name": "stories15M-q4_0.gguf",
}


class EngineError(Exception):
    pass


def _open(url, *, api=False, timeout=60):
    if not url.startswith("https://"):
        raise EngineError(f"refusing a non-HTTPS URL: {url}")
    headers = {"User-Agent": "shura-installer"}
    token = os.environ.get("GITHUB_TOKEN")
    if api and token:
        headers["Authorization"] = f"Bearer {token}"            # only ever sent to the GitHub API
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def fetch_release(tag=None, repo=None):
    """{tag, assets:[{name, url, digest}]} for `tag`, or for the newest release that has desktop builds.
    `repo` is "owner/name" of another llama.cpp fork (the TurboQuant+ prebuilts); the default is upstream."""
    base = f"https://api.github.com/repos/{repo}/releases" if repo else RELEASES
    parse = backends.parse_tqp_asset if repo == TQP_REPO else backends.parse_asset
    if tag:
        with _open(f"{base}/tags/{tag}", api=True) as r:
            releases = [json.load(r)]
    else:
        with _open(f"{base}?per_page=10", api=True) as r:
            releases = json.load(r)
    for rel in releases:
        assets = [{"name": a["name"], "url": a["browser_download_url"], "digest": a.get("digest")}
                  for a in rel.get("assets", [])]
        if any(parse(a["name"]) for a in assets):
            return {"tag": rel["tag_name"], "assets": assets}
    raise EngineError("no llama.cpp release with desktop builds was found")


def download(url, dest, sha256=None, timeout=120):
    """Download to `dest` atomically and verify the hash. Raises EngineError on a mismatch (and removes the file)."""
    dest = Path(dest)
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _open(url, timeout=timeout) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if sha256 and not backends.verify_sha256(part, sha256):
        part.unlink(missing_ok=True)
        raise EngineError(f"checksum mismatch for {url}")
    part.replace(dest)
    return dest


def extract(archive, dest):
    """Extract a .tar.gz or .zip without letting any member escape `dest`."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = str(archive)

    def safe(name):
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise EngineError(f"unsafe path in archive: {name}")

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                safe(n)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                safe(m.name)
                if m.issym() or m.islnk():
                    safe(os.path.join(os.path.dirname(m.name), m.linkname))
            t.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dest)
    return dest


def find_server(root):
    names = ("llama-server", "llama-server.exe")
    for p in sorted(Path(root).rglob("*")):
        if p.name in names and p.is_file():
            p.chmod(p.stat().st_mode | 0o111)
            return p
    raise EngineError(f"llama-server was not found in {root}")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def selftest(server, model, *, backend, timeout=120, threads=2):
    """Start the server on a tiny model, generate a few tokens, stop it. Always returns a result dict, never raises."""
    server, port = Path(server), _free_port()
    env = dict(os.environ)
    lib = str(server.parent)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = lib + os.pathsep + env.get(var, "")
    ngl = "0" if backend == "cpu" else "99"
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", "512", "-ngl", ngl,
           "-t", str(threads)]
    result = {"backend": backend, "ok": False}
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=(os.name != "nt"))
        try:
            deadline, t0 = time.monotonic() + timeout, time.monotonic()
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise EngineError("the server exited while starting")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:
                    time.sleep(0.3)
            else:
                raise EngineError(f"the server did not become healthy within {timeout} s")
            result["load_s"] = round(time.monotonic() - t0, 1)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
                                         data=json.dumps({"prompt": "Once upon a time", "n_predict": 24,
                                                          "temperature": 0, "seed": 1}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.load(r)
            tm = body.get("timings", {})
            result.update(ok=bool(body.get("content", "").strip()), tokens=tm.get("predicted_n", 0),
                          tok_s=round(tm.get("predicted_per_second", 0.0), 2),
                          prompt_tok_s=round(tm.get("prompt_per_second", 0.0), 2))
            if not result["ok"]:
                result["error"] = "the server answered with an empty text"
        except (EngineError, OSError, ValueError) as e:
            result["error"] = str(e)
        finally:
            _stop(proc)
            if not result["ok"]:
                log.seek(0)
                result["log_tail"] = log.read()[-1500:].decode(errors="replace")
    return result


def _stop(proc):
    if proc.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()


DEVICE = re.compile(r"^\s*([A-Za-z]+\d*):\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)\s*$")


def parse_devices(text):
    """`llama-server --list-devices` -> [{id, name, total, free}] in bytes. Only devices that report memory are listed."""
    out = []
    for line in (text or "").splitlines():
        m = DEVICE.match(line)
        if m:
            out.append({"id": m[1], "name": m[2], "total": int(m[3]) * 1024 ** 2, "free": int(m[4]) * 1024 ** 2})
    return out


def list_devices(server, timeout=60):
    """The GPUs a downloaded build can really use, with their memory: the final word on what hardware there is."""
    env = dict(os.environ)
    lib = str(Path(server).parent)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = lib + os.pathsep + env.get(var, "")
    try:
        r = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True, env=env, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_devices(r.stdout) if r.returncode == 0 else []


def choose_backend(results, order, tolerance=0.10):
    """Best working backend by decode speed; within `tolerance` of the best, the earlier one in `order` wins (stability:
    a backend that is only slightly faster is not worth a less-tested path). Returns (backend or None, reason)."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None, "no backend passed the self-test"
    best = max(r["tok_s"] for r in ok)
    near = [r for r in ok if r["tok_s"] >= best * (1 - tolerance)]
    near.sort(key=lambda r: order.index(r["backend"]) if r["backend"] in order else len(order))
    pick = near[0]
    why = (f"{pick['backend']} is fastest ({pick['tok_s']} tok/s)" if pick["tok_s"] == best else
           f"{pick['backend']} is within {int(tolerance * 100)}% of the fastest and is the more tested path")
    return pick["backend"], why


def smoke_model(dest_dir):
    """The pinned tiny model, downloaded once and verified."""
    path = Path(dest_dir) / SMOKE_MODEL["name"]
    if path.exists() and backends.verify_sha256(path, SMOKE_MODEL["sha256"]):
        return path
    return download(SMOKE_MODEL["url"], path, SMOKE_MODEL["sha256"])


def install_build(release, asset, dest_root, label=None, sha256=None):
    """Download, verify and extract one build. Returns the path of llama-server. The hash is `sha256` when given (a pinned
    value), else the release digest when present."""
    label = label or backends.parse_asset(asset["name"])["backend"]
    root = Path(dest_root) / f"{release['tag']}-{label}"
    try:
        return find_server(root)
    except EngineError:
        pass
    with tempfile.TemporaryDirectory(prefix="shura-dl-") as tmp:
        archive = download(asset["url"], Path(tmp) / asset["name"], sha256 or asset.get("digest"))
        extract(archive, root)
    return find_server(root)
