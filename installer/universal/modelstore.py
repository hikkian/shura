"""Downloading the model file safely: resumable, size and SHA-256 checked, atomic, with a disk-space check up front.

The size and the hash come from the Hugging Face file listing (`lfs.oid` is the SHA-256 of the file), so a corrupt or
truncated download is never used. An interrupted download is resumed from the `.part` file (HTTP Range), so a dropped
connection does not cost 18 GB again.
"""
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
SLACK = 2 * 1024 ** 3                  # disk space kept free on top of the file


class StoreError(Exception):
    pass


def _open(url, headers=None, timeout=60, allow_local=False):
    host = urllib.parse.urlparse(url).hostname
    if not url.startswith("https://") and not (allow_local and host in ("127.0.0.1", "localhost")):
        raise StoreError(f"refusing a non-HTTPS URL: {url}")
    h = {"User-Agent": "shura-installer"}
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def listing(repo, *, allow_local=False, base="https://huggingface.co"):
    """{filename: (size, sha256)} of the LFS files of a Hugging Face model repo."""
    try:
        with _open(f"{base}/api/models/{repo}/tree/main?recursive=1", allow_local=allow_local) as r:
            items = json.load(r)
    except (OSError, ValueError) as e:
        raise StoreError(f"could not read the file list of {repo}: {e}") from e
    return {i["path"]: (i["lfs"]["size"], i["lfs"]["oid"]) for i in items if i.get("lfs")}


def file_name(model, quant):
    return model["source"]["file_pattern"].format(quant=quant)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def need_bytes(dest, size):
    """Bytes still to download, counting a partial file that can be resumed."""
    part = Path(str(dest) + ".part")
    return max(0, size - (part.stat().st_size if part.exists() else 0))


def check_disk(dest, size):
    free = shutil.disk_usage(Path(dest).parent if Path(dest).parent.exists() else Path(dest).anchor or ".").free
    need = need_bytes(dest, size) + SLACK
    if free < need:
        raise StoreError(f"not enough disk space in {Path(dest).parent}: need {need / 1024 ** 3:.1f} GiB "
                         f"(the file plus {SLACK / 1024 ** 3:.0f} GiB of slack), {free / 1024 ** 3:.1f} GiB free")


def intact(dest, size, sha256):
    dest = Path(dest)
    return dest.exists() and dest.stat().st_size == size and (not sha256 or sha256_file(dest) == sha256.lower())


def fetch(url, dest, size, sha256, *, progress=None, allow_local=False):
    """Download `url` to `dest` (resuming `dest.part`), verify, move into place. Returns dest. Raises StoreError."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if intact(dest, size, sha256):
        return dest
    check_disk(dest, size)
    part = Path(str(dest) + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        try:
            r = _open(url, {"Range": f"bytes={have}-"} if have else None, timeout=120, allow_local=allow_local)
        except OSError as e:
            raise StoreError(f"download failed: {e}") from e
        with r:
            if have and r.status != 206:                       # the server ignored Range: start over
                have = 0
            done, last = have, -1
            try:
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(done * 100 / size) if size else 100
                        if progress and pct != last and pct % 2 == 0:
                            progress(done, size)
                            last = pct
            except OSError as e:
                raise StoreError(f"download interrupted at {done / 1024 ** 3:.1f} GiB: {e} "
                                 f"(run the command again, it resumes)") from e
    if part.stat().st_size != size:
        raise StoreError(f"download incomplete ({part.stat().st_size} of {size} bytes); run the command again, it resumes")
    if sha256 and sha256_file(part) != sha256.lower():
        part.replace(dest.with_suffix(dest.suffix + ".corrupt"))
        raise StoreError("the file failed its SHA-256 check and was moved aside as .corrupt; run the command again")
    os.replace(part, dest)
    return dest
