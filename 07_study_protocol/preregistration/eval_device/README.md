# A40 evaluation device pin

`EVAL_DEVICE_PIN_A40.json` is written once, with exclusive creation, after the A40 T3 GPU receipt and the two-repetition STRICT determinism receipt both pass. `v4eval.device_pin` verifies its canonical JSON self hash and the exact live GPU name and Torch/CUDA versions before the second device class is accepted. The pin's `self_sha256` appears as `device_pin_sha256` in A40 evaluation configuration, result rows, checkpoint metadata, and launch records. The historical A16 class remains accepted without a pin.

The original A16 results and new A40 results are separate comparison blocks. Re-evaluation of C_BIG and C_SMALL checkpoints on the A40 supplies the central reference rows for downstream comparisons. TEST roles stay sealed until DATA-09 provides the explicit gate.

## Format (INT-06 review, 2026-09-29; describes the file already written on the server, nothing changed)
- Writer / reader: `code/v4eval/device_pin.py` (`python -B -m v4eval.device_pin write --device-index 0 --t3-receipt <T3_GPU_RECEIPT.json> --strict-receipt <SYS04C_VARIANT_BENCH.json> --out contracts/eval_device/EVAL_DEVICE_PIN_A40.json`).
- Fields: `format` = `PPSI_V4_EVAL_DEVICE_PIN_1`, `device_class` (`NVIDIA_A40-FP32-DET`), `device_name` (exact `torch.cuda.get_device_name`), `driver` (nvidia-smi), `torch`, `cuda`, `receipts` {`t3_gpu_receipt_sha256`, `t3_verdict` = `PASS`, `strict_determinism_receipt_sha256`} (file sha256 of the two receipts), `authority` = `PREG1_AMENDMENT_06D §1`, `registered_utc`, `self_sha256`.
- `self_sha256` = sha256 of `json.dumps(pin minus self_sha256, sort_keys=True, separators=(",", ":"), allow_nan=False)` encoded UTF-8 (default ASCII escaping, so `§` is `§`; the launchers' `_canon` rule, DL 166). The file bytes are that canonical JSON of the whole pin plus a newline.
- Accepted classes = {A16-FP32-DET} ∪ {pin.device_class when the live GPU name, torch and CUDA versions equal the pin's}; a pinned class with another GPU name / torch / CUDA fails closed.
- The re-evaluation rows' sha256 cannot be a pin field (the re-evaluations need the pin): they are recorded in the DECISION_LOG row (06E §8, D5).
- Block invariant: every row of the pinned A40 block carries one `evaluator_code_sha256` (61b64e61… = the v4eval/*.py of code_ds3r2). Any byte change of `code/v4eval/*.py` splits the block (guarded by `code/tests_eval/test_device_pin.py`).
