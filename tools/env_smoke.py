from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime

import ppsi_paths

# Ensure repo root is importable when this script is invoked directly.
REPO_ROOT = ppsi_paths.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flwr
import lightgbm
import polars
import psutil
import pyarrow
import ray
import sklearn
import torch

from ppsi.training.identity import file_sha256

EXPECTED_PYTHON = (3, 11, 14)

LOCK_FILE = REPO_ROOT / "uv.lock"


def detect_nvidia_hardware() -> str:
    """Detect physical NVIDIA hardware without requiring CUDA-enabled PyTorch."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "not detected"

    if result.returncode != 0:
        return "not detected"

    output = result.stdout.strip()
    return output if output else "not detected"


def main() -> int:
    python_version = sys.version_info[:3]

    if python_version != EXPECTED_PYTHON:
        expected = ".".join(map(str, EXPECTED_PYTHON))

        print(
            "SMOKE STATUS: FAIL\n"
            f"Unsupported Python version. Expected {expected}, "
            f"got {platform.python_version()}."
        )
        return 1

    if not LOCK_FILE.is_file():
        print(f"SMOKE STATUS: FAIL\nMissing lock file: {LOCK_FILE}")
        return 1

    cuda_available = torch.cuda.is_available()

    device = torch.device("cuda" if cuda_available else "cpu")

    left = torch.tensor(
        [1.0, 2.0],
        device=device,
    )

    right = torch.tensor(
        [3.0, 4.0],
        device=device,
    )

    result = left + right

    expected_result = torch.tensor(
        [4.0, 6.0],
        device=device,
    )

    if not torch.equal(result, expected_result):
        print("SMOKE STATUS: FAIL\nTiny tensor operation failed.")
        return 1

    print("=== Training Environment Smoke ===")
    print(f"timestamp_utc: {datetime.now(UTC).isoformat()}")
    print(f"os: {platform.platform()}")
    print(f"architecture: {platform.machine()}")
    print(f"python: {platform.python_version()}")
    print(f"python_executable: {sys.executable}")
    print(f"uv_lock_sha256: {file_sha256(LOCK_FILE)}")

    print()
    print("=== Core Package Versions ===")
    print(f"torch: {torch.__version__}")
    print(f"flower: {flwr.__version__}")
    print(f"ray: {ray.__version__}")
    print(f"polars: {polars.__version__}")
    print(f"pyarrow: {pyarrow.__version__}")
    print(f"psutil: {psutil.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"lightgbm: {lightgbm.__version__}")

    print()
    print("=== Compute Identity ===")
    print(f"torch_cuda_runtime: {torch.version.cuda}")
    print(f"torch_cuda_available: {cuda_available}")
    print(f"nvidia_hardware: {detect_nvidia_hardware()}")
    print(f"selected_device: {device}")

    if cuda_available:
        print(f"torch_cuda_device: {torch.cuda.get_device_name(0)}")
    else:
        print("cpu_fallback: active")

    print()
    print(f"tensor_result: {result.cpu().tolist()}")
    print("SMOKE STATUS: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
