"""v4eval.frozen_config — the frozen T3 evaluation numerics (contracts/PREG1_LOCK_VALUES.json, section T3_eval_numerics).

T3 is binding for C, FA, FP, PF, CPF, LO, in-training marks, final rows, the tie-break and cold start:
  score_block 158486 (= K: one block, so the scores are BaseAdapter.logits bit-for-bit), class_chunk 32768,
  eval_batch 1024 in manifest order (strictly ascending decision_id, tail batch unpadded), cache_bytes 1073741824,
  FP64 metric accumulation, strict FP32 with deterministic flags, comparison-key additions eval_score_block /
  eval_batch / eval_device_class, a per-row tie diagnostic (n_tied and pessimistic ranks).
Memory rule: under GPU memory contention ONLY eval_batch may change, through a recorded amendment; never score_block.

`load()` reads the lock JSON, checks every T3 value against the values frozen here (a lock file whose T3 values
differ is refused), records the file's sha256, and returns an immutable FrozenEvalConfig. It refuses every caller
override; the one permitted change is an EvalBatchAmendment object (a recorded amendment, never a dict or kwarg).
A FrozenEvalConfig cannot be constructed except by load(), and cannot be mutated.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

import torch

from .errors import V4EvalError
from .hashing import file_sha256

DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[2] / "contracts" / "PREG1_LOCK_VALUES.json"
SECTION = "T3_eval_numerics"
FROZEN_DEVICE_CLASS = "A16-FP32-DET"
CPU_DEVICE_CLASS = "CPU-FP32-DET"
METHODS = ("C", "FA", "FP", "PF", "CPF", "LO")
# R2 (G1 combined gate): the frozen config's own contract name for the evaluation stream. PREG1_LOCK_VALUES.json (a
# contracts/* file) carries no such key and is never edited to add one, so this is a fixed code-level value, not read
# from the lock JSON; ALL_ROWS is refused (the only value a frozen config may ever carry is EVAL_ROWS_ONLY). v4eval's
# own internal stream name (E2E_ROWS; v4int.providers.STREAM_NAMES) stays internal — the mapping is recorded, never
# exposed as the contract name.
EVAL_STREAM = "EVAL_ROWS_ONLY"
EVAL_STREAM_INTERNAL_MAPPING = "E2E_ROWS"

EXPECTED_HEADER = {"document": "PREG1_LOCK_VALUES", "version": "1.0.0", "status": "FROZEN_BEFORE_ANY_CENTRAL_RESULT"}
EXPECTED_T3 = {
    "applies_to": ["C", "FA", "FP", "PF", "CPF", "LO", "in-training marks", "final rows", "tie-break", "cold-start"],
    "score_block": 158486,
    "class_chunk": 32768,
    "eval_batch": 1024,
    "order": "manifest order, strictly ascending decision_id; tail batch unpadded",
    "cache_bytes": 1073741824,
    "device": "A16 CUDA, strict FP32",
    "flags": {"matmul.allow_tf32": False, "cudnn.allow_tf32": False, "cudnn.benchmark": False,
              "cudnn.deterministic": True, "use_deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    "metric_accumulation": "FP64",
    "comparison_key_additions": {"eval_score_block": 158486, "eval_batch": 1024, "eval_device_class": FROZEN_DEVICE_CLASS},
    "per_row_tie_diagnostic": "n_tied and pessimistic ranks",
}


class FrozenConfigError(V4EvalError):
    """A caller override, a tampered or mismatched lock, a non-frozen artifact, or runtime flags off the frozen values."""


_TOKEN = object()


@dataclass(frozen=True)
class EvalBatchAmendment:
    """The only permitted change to T3: a recorded eval_batch reduction under GPU memory contention (T3 memory_rule)."""
    amendment_id: str
    eval_batch: int
    reason: str
    recorded_utc: str
    recorded_by: str
    record_path: Optional[str] = None
    record_sha256: Optional[str] = None

    def validate(self, frozen_eval_batch: int) -> None:
        if not self.amendment_id or not self.reason or not self.recorded_by:
            raise FrozenConfigError("an eval_batch amendment needs amendment_id, reason and recorded_by")
        if not isinstance(self.eval_batch, int) or isinstance(self.eval_batch, bool):
            raise FrozenConfigError("amended eval_batch must be an int")
        if not (1 <= self.eval_batch < frozen_eval_batch):
            raise FrozenConfigError(f"amended eval_batch must be in [1, {frozen_eval_batch}) (a memory-contention "
                                    "reduction)")
        try:
            _dt.datetime.fromisoformat(self.recorded_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise FrozenConfigError(f"recorded_utc must be ISO-8601: {self.recorded_utc!r}") from e
        if (self.record_path is None) != (self.record_sha256 is None):
            raise FrozenConfigError("record_path and record_sha256 go together")
        if self.record_path is not None:
            p = Path(self.record_path)
            if not p.is_file() or file_sha256(p) != self.record_sha256:
                raise FrozenConfigError(f"amendment record {p} is missing or its sha256 does not match")


@dataclass(frozen=True)
class FrozenEvalConfig:
    score_block: int
    class_chunk: int
    eval_batch: int
    cache_bytes: int
    metric_accumulation: str
    device_class: str
    order: str
    flags: Mapping
    applies_to: tuple
    lock_path: str
    lock_sha256: str
    lock_version: str
    t3_sha256: str = ""
    eval_stream: str = EVAL_STREAM
    eval_stream_source: Mapping = field(default_factory=lambda: MappingProxyType({
        "contract_name": EVAL_STREAM, "internal_mapping": EVAL_STREAM_INTERNAL_MAPPING,
        "rule": "R2 (G1 combined gate): v4eval's E2E_ROWS mapping stays internal; ALL_ROWS is refused",
        "all_rows_refused": True}))
    amendment: Optional[EvalBatchAmendment] = None
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _TOKEN:
            raise FrozenConfigError("FrozenEvalConfig is built only by v4eval.frozen_config.load()")

    @property
    def amendment_id(self) -> Optional[str]:
        return None if self.amendment is None else self.amendment.amendment_id

    # the attribute names of the v4train test stub (tests_train/frozen_config_stub.py), read by EvalConfig.from_frozen
    @property
    def eval_device_class(self) -> str:
        return self.device_class

    @property
    def comparison_key_additions(self) -> dict:
        return {"eval_score_block": self.score_block, "eval_batch": self.eval_batch,
                "eval_device_class": self.device_class}

    @property
    def source(self) -> str:
        return "v4eval.frozen_config.load() (contracts/PREG1_LOCK_VALUES.json T3_eval_numerics)"

    @property
    def sha256(self) -> str:
        """sha256 of the T3 section (the stub's definition); an amendment is folded in, so it changes the identity."""
        if self.amendment is None:
            return self.t3_sha256
        a = json.dumps({"t3_sha256": self.t3_sha256, "amendment_id": self.amendment.amendment_id,
                        "eval_batch": self.amendment.eval_batch}, sort_keys=True)
        return hashlib.sha256(a.encode("utf-8")).hexdigest()

    def eval_config(self, path: str, device_class: str) -> dict:
        """The artifact / result eval_config for this frozen configuration. `eval_stream` here is the frozen
        CONTRACT name (INT-02R1 F6); v4eval.predict's own stream cursor still overwrites this key with the ACTUAL
        observed form (E2E_ROWS / ALL_ROWS — a batch-composition fact, not a frozen constant: v4train's historical
        ALL_ROWS callers must keep working), so this is an informational default only, never re-verified by
        check_artifact_config (FROZEN_FIELDS does not name it: ALL_ROWS is a legitimate artifact there, refused only
        at the R4 comparison-key layer, v4int.providers._refuse_non_frozen_eval_stream)."""
        return {"frozen": True, "path": path, "score_block": self.score_block, "class_chunk": self.class_chunk,
                "eval_batch": self.eval_batch, "cache_bytes": self.cache_bytes,
                "metric_accumulation": self.metric_accumulation, "eval_device_class": device_class,
                "device_class_frozen": self.device_class, "lock_sha256": self.lock_sha256,
                "lock_version": self.lock_version, "eval_batch_amendment_id": self.amendment_id,
                "eval_stream": self.eval_stream}


FROZEN_FIELDS = ("frozen", "score_block", "class_chunk", "eval_batch", "cache_bytes", "metric_accumulation",
                 "device_class_frozen", "lock_sha256", "lock_version", "eval_batch_amendment_id")


def _verify(doc: Mapping) -> dict:
    for k, v in EXPECTED_HEADER.items():
        if doc.get(k) != v:
            raise FrozenConfigError(f"lock {k} is {doc.get(k)!r}, expected {v!r}")
    t3 = doc.get(SECTION)
    if not isinstance(t3, Mapping):
        raise FrozenConfigError(f"lock has no {SECTION} section")
    for k, v in EXPECTED_T3.items():
        if t3.get(k) != v:
            raise FrozenConfigError(f"T3 {k} is {t3.get(k)!r}, frozen value is {v!r}")
    if t3["score_block"] != t3["comparison_key_additions"]["eval_score_block"] or \
            t3["eval_batch"] != t3["comparison_key_additions"]["eval_batch"]:
        raise FrozenConfigError("T3 comparison_key_additions disagree with the T3 values")
    return dict(t3)


def load(path: Optional[Path] = None, *, amendment: Optional[EvalBatchAmendment] = None, **overrides) -> FrozenEvalConfig:
    """The frozen T3 configuration. Every override is refused; only a recorded EvalBatchAmendment may change eval_batch."""
    if overrides:
        raise FrozenConfigError(f"override refused for {sorted(overrides)}: T3 is frozen; only a recorded "
                                "EvalBatchAmendment may change eval_batch, never score_block")
    if amendment is not None and not isinstance(amendment, EvalBatchAmendment):
        raise FrozenConfigError("an amendment must be an EvalBatchAmendment object (a recorded amendment), "
                                f"not {type(amendment).__name__}")
    p = Path(path) if path is not None else DEFAULT_LOCK_PATH
    if not p.is_file():
        raise FrozenConfigError(f"lock file {p} not found")
    raw = p.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except ValueError as e:
        raise FrozenConfigError(f"lock file {p} is not strict JSON: {e}") from e
    t3 = _verify(doc)
    eval_batch = int(t3["eval_batch"])
    if amendment is not None:
        amendment.validate(eval_batch)
        eval_batch = amendment.eval_batch
    # R2: the fingerprint is recomputed to fold in the frozen eval_stream contract name (a fixed code-level value,
    # not part of doc[SECTION]; see EVAL_STREAM above), so a code change to it changes the identity too.
    t3_sha256 = hashlib.sha256(json.dumps(dict(t3, eval_stream=EVAL_STREAM), sort_keys=True).encode("utf-8")).hexdigest()
    return FrozenEvalConfig(score_block=int(t3["score_block"]), class_chunk=int(t3["class_chunk"]),
                            eval_batch=eval_batch, cache_bytes=int(t3["cache_bytes"]),
                            metric_accumulation=str(t3["metric_accumulation"]),
                            device_class=t3["comparison_key_additions"]["eval_device_class"], order=str(t3["order"]),
                            flags=MappingProxyType(dict(t3["flags"])), applies_to=tuple(t3["applies_to"]),
                            lock_path=str(p), lock_sha256=file_sha256(p), lock_version=str(doc["version"]),
                            t3_sha256=t3_sha256, eval_stream=EVAL_STREAM,
                            amendment=amendment, _token=_TOKEN)


def require_frozen(cfg) -> FrozenEvalConfig:
    """Every public evaluate path calls this: the config must come from load() and the lock file must be unchanged."""
    if not isinstance(cfg, FrozenEvalConfig) or cfg._token is not _TOKEN:
        raise FrozenConfigError("a FrozenEvalConfig from v4eval.frozen_config.load() is required")
    if file_sha256(Path(cfg.lock_path)) != cfg.lock_sha256:
        raise FrozenConfigError("the lock file changed after the frozen config was loaded")
    return cfg


def resolve(config=None, **given) -> FrozenEvalConfig:
    """The frozen config for a public evaluate path: `config` (from load()) or, if None, load(). Any numeric a caller
    passes (score_block, class_chunk, cache_bytes, eval_batch) must EQUAL the frozen value; otherwise it is an
    override and is refused (T3: only a recorded EvalBatchAmendment may change eval_batch)."""
    cfg = require_frozen(config) if config is not None else load()
    bad = {k: v for k, v in given.items() if v is not None and int(v) != int(getattr(cfg, k))}
    if bad:
        raise FrozenConfigError(f"override refused: {bad} differ from the frozen T3 values "
                                f"{ {k: getattr(cfg, k) for k in bad} }")
    return cfg


def apply_runtime_flags(cfg: FrozenEvalConfig) -> None:
    """Set the frozen T3 flags (for launchers; CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes)."""
    require_frozen(cfg)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cfg.flags["CUBLAS_WORKSPACE_CONFIG"])
    torch.backends.cuda.matmul.allow_tf32 = cfg.flags["matmul.allow_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg.flags["cudnn.allow_tf32"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = cfg.flags["cudnn.benchmark"]
    torch.backends.cudnn.deterministic = cfg.flags["cudnn.deterministic"]
    torch.use_deterministic_algorithms(cfg.flags["use_deterministic_algorithms"], warn_only=False)


def check_runtime_flags(cfg: FrozenEvalConfig, device: torch.device) -> None:
    bad = []
    f = cfg.flags
    if bool(torch.backends.cuda.matmul.allow_tf32) != f["matmul.allow_tf32"]:
        bad.append("torch.backends.cuda.matmul.allow_tf32")
    if bool(torch.backends.cudnn.allow_tf32) != f["cudnn.allow_tf32"]:
        bad.append("torch.backends.cudnn.allow_tf32")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append("float32_matmul_precision != highest")
    if bool(torch.backends.cudnn.benchmark) != f["cudnn.benchmark"]:
        bad.append("torch.backends.cudnn.benchmark")
    if bool(torch.backends.cudnn.deterministic) != f["cudnn.deterministic"]:
        bad.append("torch.backends.cudnn.deterministic")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        bad.append("use_deterministic_algorithms(True, warn_only=False)")
    if torch.device(device).type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != f["CUBLAS_WORKSPACE_CONFIG"]:
        bad.append("CUBLAS_WORKSPACE_CONFIG")
    if bad:
        raise FrozenConfigError("runtime flags differ from the frozen T3 values: " + ", ".join(bad))


def eval_device_class(device: torch.device) -> str:
    """A16-FP32-DET on an A16 (the frozen class); otherwise a distinct class, so rows never mix devices."""
    d = torch.device(device)
    if d.type == "cuda":
        name = torch.cuda.get_device_name(d)
        return FROZEN_DEVICE_CLASS if "A16" in name else "".join(c if c.isalnum() else "_" for c in name) + "-FP32-DET"
    return CPU_DEVICE_CLASS if d.type == "cpu" else f"{d.type.upper()}-FP32-DET"


def check_artifact_config(cfg: FrozenEvalConfig, eval_config: Mapping) -> None:
    """The artifact must have been produced under exactly this frozen configuration."""
    want = cfg.eval_config(path="*", device_class="*")
    diff = [k for k in FROZEN_FIELDS if (eval_config or {}).get(k) != want[k]]
    if diff:
        raise FrozenConfigError(f"artifact eval_config differs from the frozen config in {diff}")
