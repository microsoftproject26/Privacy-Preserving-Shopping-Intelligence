"""Drive the Lightning studio from here: push code, run, pull results back.

The rule this enforces is the one that matters: **the studio is where things run, this
machine is where they live.** Every artifact comes home as soon as it exists, so a studio
that sleeps, is interrupted, or gets recycled costs compute and never costs work.

Three transfer facts shaped it, all measured rather than assumed:

* `upload_folder` writes Windows path separators into the *filenames*, so a directory tree
  arrives as flat files called ``a\\b\\c.py``. Everything moves as a zip instead.
* The link runs at about **0.4 MB/s**. Sending the caches raw - 3.6 GB - would take over
  two hours.
* Those caches compress **14 to 19 times**, because a window is 20 slots holding a median
  of 7 real events and the rest is constant padding. 3.6 GB becomes 247 MB, and the
  transfer becomes minutes.

Usage:

    python studio.py status
    python studio.py push  S2-DS-06_T2_Purchase_Head
    python studio.py run   "cd ~/s2ds01 && python -u ladder.py"
    python studio.py pull  S2-DS-06_T2_Purchase_Head
    python studio.py gpu   L40S
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
REMOTE = "s2ds01"

STUDIO = {"name": "project", "teamspace": "visual-content-ai-project", "org": "aedeid0-org"}

# The credentials live in the environment, never in this file and never in git.
for required in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY"):
    if not os.environ.get(required):
        raise SystemExit(
            f"{required} is not set.\n\n"
            "  export LIGHTNING_USER_ID=...\n"
            "  export LIGHTNING_API_KEY=...\n\n"
            "or run:  .venv/Scripts/lightning login"
        )


def connect():
    from lightning_sdk import Studio

    studio = Studio(**STUDIO)
    studio.show_progress = False
    return studio


def push(folder: str) -> None:
    """Zip a local folder, upload it, unpack it under the remote working directory."""
    source = PROJECT / folder
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")

    skip = {"cache", "cache_t2", "cache_t3", "output", "__pycache__"}
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "push.zip"
        count = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(source.rglob("*")):
                if not path.is_file() or set(path.relative_to(source).parts) & skip:
                    continue
                # Forward slashes explicitly: the archive is unpacked on Linux.
                zf.write(path, path.relative_to(source).as_posix())
                count += 1
        size = archive.stat().st_size
        print(f"  {count} files -> {size / 1e6:.2f} MB")

        studio = connect()
        started = time.time()
        studio.upload_file(str(archive), remote_path="push.zip")
        print(f"  uploaded in {time.time() - started:.0f}s")
        print(
            studio.run(
                f"cd ~/{REMOTE} && rm -rf {folder} && mkdir -p {folder} && "
                f"unzip -qo ~/push.zip -d {folder} && rm ~/push.zip && "
                f"echo unpacked $(find {folder} -type f | wc -l) files"
            )
        )


def pull(folder: str) -> None:
    """Bring a remote task's output home, so the results outlive the studio."""
    studio = connect()
    remote = f"~/{REMOTE}/{folder}/output"
    listing = studio.run(
        f"test -d {remote} && find {remote} -type f | sed 's|.*/||' || echo MISSING"
    )
    if "MISSING" in listing:
        raise SystemExit(f"no output directory at {remote}")

    target = PROJECT / folder / "output"
    target.mkdir(parents=True, exist_ok=True)
    names = [line.strip() for line in listing.splitlines() if line.strip()]
    print(f"  {len(names)} files to fetch")

    studio.run(f"cd ~/{REMOTE}/{folder} && rm -f ~/pull.zip && zip -qr ~/pull.zip output")
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "pull.zip"
        started = time.time()
        studio.download_file("pull.zip", str(archive))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(PROJECT / folder)
        print(f"  pulled {archive.stat().st_size / 1e6:.2f} MB in {time.time() - started:.0f}s")
    for path in sorted(target.iterdir()):
        print(f"    {path.stat().st_size / 1e6:8.2f} MB  {path.name}")


def run(command: str) -> None:
    studio = connect()
    print(studio.run(command))


def status() -> None:
    studio = connect()
    print(f"  studio      : {studio.name}")
    print(f"  status      : {studio.status}")
    print(f"  machine     : {studio.machine}")
    print(f"  interruptible: {studio.interruptible}")
    print()
    print(
        studio.run(
            f"cd ~/{REMOTE} 2>/dev/null && echo '--- remote tree ---' && ls && "
            "echo && echo '--- data ---' && du -sh data/* 2>/dev/null && "
            "echo && echo '--- gpu ---' && (nvidia-smi --query-gpu=name,memory.total "
            "--format=csv,noheader 2>/dev/null || echo 'no GPU attached') && "
            'echo && python -c \'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())\''
        )
    )


def gpu(machine: str) -> None:
    """Switch the machine, then reinstall torch for whatever is now attached.

    A CPU torch build on a GPU machine reports `cuda: False` and silently trains on the
    CPU at a fortieth of the speed. Switching without reinstalling is the quiet version of
    that mistake.
    """
    studio = connect()
    print(f"  switching {studio.machine} -> {machine}")
    studio.switch_machine(machine)
    print(f"  now: {studio.machine}, status {studio.status}")
    print(
        studio.run(
            "pip install -q torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2; "
            'python -c \'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())\'; '
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        )
    )


COMMANDS = {"push": push, "pull": pull, "run": run, "status": status, "gpu": gpu}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python studio.py [{' | '.join(COMMANDS)}] [argument]")
    action = COMMANDS[sys.argv[1]]
    action(*sys.argv[2:]) if len(sys.argv) > 2 else action()
