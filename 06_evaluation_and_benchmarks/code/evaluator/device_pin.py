"""Registered second evaluation device class (PREG1 amendment 06D §1)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


FORMAT = "PPSI_V4_EVAL_DEVICE_PIN_1"
DEFAULT_PIN = Path(__file__).resolve().parents[2] / "contracts" / "eval_device" / "EVAL_DEVICE_PIN_A40.json"


class DevicePinError(RuntimeError):
    pass


def _canon(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(obj: dict) -> str:
    return hashlib.sha256(_canon(obj)).hexdigest()


def load_pin(path: Path | str = DEFAULT_PIN) -> dict:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("format") != FORMAT or obj.get("authority") != "PREG1_AMENDMENT_06D §1":
        raise DevicePinError("unknown evaluation device pin format or authority")
    required = ("device_class", "device_name", "driver", "torch", "cuda", "receipts", "registered_utc")
    if any(not obj.get(k) for k in required):
        raise DevicePinError("incomplete evaluation device pin")
    receipts = obj["receipts"]
    if not isinstance(receipts, dict) or receipts.get("t3_verdict") != "PASS" or any(
        not receipts.get(k) for k in ("t3_gpu_receipt_sha256", "strict_determinism_receipt_sha256")
    ):
        raise DevicePinError("evaluation device pin lacks passing receipts")
    saved = obj.get("self_sha256")
    if not isinstance(saved, str) or saved != _digest({k: v for k, v in obj.items() if k != "self_sha256"}):
        raise DevicePinError("evaluation device pin self hash mismatch")
    return obj


def active_pin(path: Path | str = DEFAULT_PIN) -> dict | None:
    return load_pin(path) if Path(path).is_file() else None


def pin_for_device(device: torch.device | str, path: Path | str = DEFAULT_PIN) -> dict | None:
    """Return a matching pin; fail closed when a registered class has the wrong physical device."""
    pin = active_pin(path)
    if pin is None:
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    from .frozen_config import eval_device_class
    cls = eval_device_class(dev)
    if cls != pin["device_class"]:
        return None
    if torch.cuda.get_device_name(dev) != pin["device_name"]:
        raise DevicePinError("live GPU name differs from the registered device pin")
    if torch.__version__ != pin["torch"] or torch.version.cuda != pin["cuda"]:
        raise DevicePinError("live Torch/CUDA differs from the registered device pin")
    return pin


def accepted_classes(frozen_class: str, pin: dict | None) -> set[str]:
    return {frozen_class} | ({pin["device_class"]} if pin else set())


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pin(*, device_index: int, t3_receipt: Path, strict_receipt: Path,
              out: Path = DEFAULT_PIN, driver: str | None = None) -> dict:
    """Write once after two independently generated passing A40 receipts exist."""
    t3 = json.loads(Path(t3_receipt).read_text(encoding="utf-8"))
    strict = json.loads(Path(strict_receipt).read_text(encoding="utf-8"))
    if t3.get("status") != "COMPLETE" or t3.get("all_checks_pass") is not True:
        raise DevicePinError("T3 receipt did not pass")
    if set(t3.get("families", {})) != {"GRU", "SASREC"} or not all(
        f.get("checks_pass") is True and not f.get("a16_reference", {}).get("missing")
        for f in t3["families"].values()
    ):
        raise DevicePinError("T3 receipt lacks the two family checks and A16 comparison")
    if strict.get("all_checks_pass") is not True:
        raise DevicePinError("STRICT determinism receipt did not pass")
    variants = {r.get("variant"): r for r in strict.get("table", [])}
    if set(variants) != {"SASREC_D64_B2", "SASREC_D256_B3"} or any(
        r.get("n_reps") != 2 or r.get("bitwise_rep_to_rep") is not True for r in variants.values()
    ):
        raise DevicePinError("STRICT receipt lacks two identical reps for both registered variants")
    dev = torch.device(f"cuda:{int(device_index)}")
    from .frozen_config import eval_device_class
    cls = eval_device_class(dev)
    name = torch.cuda.get_device_name(dev)
    if cls != "NVIDIA_A40-FP32-DET" or t3.get("device", {}).get("eval_device_class") != cls:
        raise DevicePinError("T3 receipt and live device are not the registered A40 class")
    if t3.get("device", {}).get("name") != name:
        raise DevicePinError("T3 receipt GPU name differs from live GPU")
    if driver is None:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(device_index)],
            text=True).strip()
    obj = {"format": FORMAT, "device_class": cls, "device_name": name,
           "driver": driver, "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "receipts": {"t3_gpu_receipt_sha256": _file_sha(Path(t3_receipt)), "t3_verdict": "PASS",
                        "strict_determinism_receipt_sha256": _file_sha(Path(strict_receipt))},
           "authority": "PREG1_AMENDMENT_06D §1",
           "registered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    obj["self_sha256"] = _digest(obj)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_canon(obj) + b"\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write"])
    ap.add_argument("--device-index", type=int, required=True)
    ap.add_argument("--t3-receipt", type=Path, required=True)
    ap.add_argument("--strict-receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_PIN)
    args = ap.parse_args()
    obj = write_pin(device_index=args.device_index, t3_receipt=args.t3_receipt,
                    strict_receipt=args.strict_receipt, out=args.out)
    print(obj["self_sha256"])


if __name__ == "__main__":
    main()
