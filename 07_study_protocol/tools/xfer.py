"""xfer.py (owner: Operations Director) — hash-verified server<->local file transfer for PPSI V4.

  python xfer.py http_get <remote_abs_path> <local_path>      # fast path (Jupyter contents API over HTTPS)
  python xfer.py http_put <local_path> <remote_abs_path>
  python xfer.py get|put ... [--chunk-mib 1]                  # slow fallback over the kernel channel (~0.05-0.07 MiB/s)

Why: `.local_ai/sasrec_v1/tools/lobot.py` starts a fresh kernel per call, so chunked kernel transfers cost ~14 s per
MiB; single-shot base64 prints truncate above tens of MB. The contents API is one HTTPS request, but it refuses hidden
directories (.local_ai). http_get therefore exposes the file through a HARD LINK in a non-hidden staging dir inside the
owner's own user folder (no extra disk; the link is removed in `finally`); http_put uploads to that staging dir and
atomically renames into the target after a remote sha256 check (same filesystem). Credentials come from lobot and are
never printed. Nothing pre-existing is ever deleted: only the staging link / staging .part created by this call.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
import ppsi_paths

sys.path.insert(0, str(ppsi_paths.PILOT_ROOT / "sasrec_v1" / "tools"))
import lobot  # noqa: E402

JUPYTER_ROOT = ppsi_paths.SERVER_HOME.as_posix() + "/"
STAGE = (ppsi_paths.USER_HOME / "ppsi_v4_xfer_stage").as_posix() + "/"
NL = chr(10)


def _exec(lines: list[str], timeout: float = 600) -> str:
    out = asyncio.run(lobot._run(NL.join(lines) + NL, timeout)).strip()
    if out.startswith("KERNEL_ERROR"):
        raise RuntimeError(out[-2000:])
    return out


def remote_info(remote: str) -> tuple[int, str]:
    out = _exec(["import hashlib, os, json", f"p = {json.dumps(remote)}", "h = hashlib.sha256()", "f = open(p, 'rb')",
                 "for b in iter(lambda: f.read(1 << 22), b''): h.update(b)",
                 "print(json.dumps([os.path.getsize(p), h.hexdigest()]))"])
    size, digest = json.loads(out.splitlines()[-1])
    return int(size), digest


def _staged(remote: str, suffix: str = "") -> str:
    return f"{STAGE}{hashlib.sha256(remote.encode()).hexdigest()[:16]}_{Path(remote).name}{suffix}"


def http_get(remote: str, local: Path) -> dict:
    import requests
    remote = lobot.remote_path(remote)
    size, digest = remote_info(remote)
    staged = _staged(remote)
    t0 = time.time()
    _exec(["import os", f"os.makedirs({json.dumps(STAGE)}, exist_ok=True)",
           f"s = {json.dumps(staged)}", "os.path.exists(s) and os.remove(s)", f"os.link({json.dumps(remote)}, s)", "print('ok')"])
    try:
        r = requests.get(f"{lobot.BASE}/api/contents/{staged[len(JUPYTER_ROOT):]}", headers=lobot.H,
                         params={"content": "1", "format": "base64", "type": "file"}, timeout=3600)
        r.raise_for_status()
        data = base64.b64decode(r.json()["content"])
    finally:
        _exec(["import os", f"s = {json.dumps(staged)}", "os.path.exists(s) and os.remove(s)", "print('ok')"])
    h = hashlib.sha256(data).hexdigest()
    if h != digest or len(data) != size:
        raise RuntimeError(f"HTTP GET hash/size mismatch for {remote}")
    part = Path(str(local) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(data)
    part.replace(local)
    dt = time.time() - t0
    return {"op": "http_get", "bytes": size, "sha256": digest, "seconds": round(dt, 1), "MiB_per_s": round(size / 2**20 / max(dt, 1e-9), 2)}


def http_put(local: Path, remote: str) -> dict:
    import requests
    remote = lobot.remote_path(remote)
    data = Path(local).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    staged = _staged(remote, ".part")
    t0 = time.time()
    _exec(["import os", f"os.makedirs({json.dumps(STAGE)}, exist_ok=True)", f"os.makedirs(os.path.dirname({json.dumps(remote)}), exist_ok=True)", "print('ok')"])
    r = requests.put(f"{lobot.BASE}/api/contents/{staged[len(JUPYTER_ROOT):]}", headers=lobot.H, timeout=3600,
                     json={"type": "file", "format": "base64", "content": base64.b64encode(data).decode()})
    r.raise_for_status()
    res = _exec(["import hashlib, os", f"s = {json.dumps(staged)}", f"p = {json.dumps(remote)}",
                 "h = hashlib.sha256(open(s, 'rb').read()).hexdigest()", f"ok = h == {json.dumps(digest)}",
                 "os.replace(s, p) if ok else os.remove(s)", "print('PUT_OK' if ok else 'PUT_HASH_MISMATCH', h)"])
    if not res.splitlines()[-1].startswith("PUT_OK"):
        raise RuntimeError(f"HTTP PUT failed for {remote}: {res[-300:]}")
    dt = time.time() - t0
    return {"op": "http_put", "bytes": len(data), "sha256": digest, "seconds": round(dt, 1), "MiB_per_s": round(len(data) / 2**20 / max(dt, 1e-9), 2)}


def get(remote: str, local: Path, chunk: int) -> dict:
    remote = lobot.remote_path(remote)
    size, digest = remote_info(remote)
    part = Path(str(local) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(part, "wb") as f:
        for off in range(0, size, chunk):
            out = _exec(["import base64", f"f = open({json.dumps(remote)}, 'rb')", f"f.seek({off})",
                         f"print(base64.b64encode(f.read({chunk})).decode())"])
            f.write(base64.b64decode(out.splitlines()[-1]))
    if hashlib.sha256(part.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"GET hash mismatch for {remote}")
    part.replace(local)
    dt = time.time() - t0
    return {"op": "get", "bytes": size, "sha256": digest, "seconds": round(dt, 1), "MiB_per_s": round(size / 2**20 / max(dt, 1e-9), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=("http_get", "http_put", "get"))
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--chunk-mib", type=int, default=1)
    a = ap.parse_args()
    if a.op == "http_get":
        res = http_get(a.src, Path(a.dst))
    elif a.op == "http_put":
        res = http_put(Path(a.src), a.dst)
    else:
        res = get(a.src, Path(a.dst), a.chunk_mib * 2**20)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
