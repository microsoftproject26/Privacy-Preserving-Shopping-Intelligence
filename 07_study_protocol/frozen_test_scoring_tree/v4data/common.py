"""v4data.common (owner: Data Lead, task DATA-04) - paths, constants, hashing, atomic IO, safe paths, identity keys.

Everything here is data-only and model-free. No raw identifier is ever written by any function in this module;
the identity helpers turn a raw user_id string into the salted pseudonym of CLIENT_CONTRACT_V1 C1.2 in memory.

Hashing follows MANIFEST_SPEC_V4.json global_rules.hashing exactly (table_content_sha256, ordered decision-id
hashes, population_sha256, user_set_sha256, candidate_order_hash).
Atomicity follows global_rules.file_format.atomicity and PKG/12 T44-T46: write to a temporary file in the same
directory, fsync, verify, rename; success markers are written last and only after verification.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# ----------------------------------------------------------------------------------------------- paths
V4 = Path(__file__).resolve().parents[2]               # .../.local_ai/ppsi_v4
CODE = V4 / "code"
V4DATA = CODE / "v4data"
FROZEN_INPUTS = V4DATA / "frozen_inputs"
SERVER_DATASET = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
LOCAL_DATASET = Path("F:/Queens University/Microsoft Project/eda/rees46_nextitem_v1")


def dataset_root() -> Path:
    """The frozen v1 dataset root: $V4_DATASET_ROOT, else the server copy if present, else the local copy."""
    env = os.environ.get("V4_DATASET_ROOT")
    if env:
        return Path(env)
    return SERVER_DATASET if SERVER_DATASET.is_dir() else LOCAL_DATASET


def data_v2_root() -> Path:
    """Owned output root (DATA-04 packet): $V4_DATA_V2_ROOT, else ppsi_v4/data_v2."""
    env = os.environ.get("V4_DATA_V2_ROOT")
    return Path(env) if env else V4 / "data_v2"


# ------------------------------------------------------------------------------------------- constants
VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"    # V1/VERSION.json:3
DATASET_CODE_HASH = "02fbb07147ed9c0ba54092dd4ee042ce6af42a06145178017999fb7b242dde73"  # V1/VERSION.json:20
SCHEMA_VERSION = "2.1.0"
SCHEMA_SELF_SHA256 = "55a244e1608bcdcdf6f1728595f534871b52154faebf2977e11ac1fb487052f9"
SALT = b"PPSI_V4_POPULATION_20260926"
SALT_STR = "PPSI_V4_POPULATION_20260926"
K = 158_486
N_BUCKETS = 64
PAD, MISSING, OOV, KNOWN_START = 0, 1, 2, 3

NON_TEST_SPLITS = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")
VALIDATION_SPLITS = ("VALIDATION_SELECT", "VALIDATION_CONFIRM")
TEST_SPLIT_NAMES = ("TEST", "RETRO_TEST")


def utc_us(iso: str) -> int:
    """int64 microseconds since 1970-01-01T00:00:00Z of a UTC ISO timestamp (no tz inference)."""
    d = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    delta = d - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


# EVALUATION_CONTRACT_V4 section 1.1 (PKG/03 section 2 L10-L13); all half-open [lo, hi)
TRAIN_START_US = utc_us("2019-10-01T00:00:00Z")
TRAIN_END_US = utc_us("2019-11-16T00:00:00Z")
VS_END_US = utc_us("2019-11-20T00:00:00Z")            # VAL_A / VAL_B subperiod boundary
VALIDATION_END_US = utc_us("2019-11-23T00:00:00Z")
TEST_END_US = utc_us("2019-11-30T00:00:00Z")
GRACE_END_US = utc_us("2019-12-01T00:00:00Z")


def v4_split_of(ts_us: np.ndarray) -> np.ndarray:
    """V4 split code per decision_ts (half-open). 0 TRAIN, 1 VALIDATION, 2 RETRO_TEST, -1 outside/GRACE."""
    ts = np.asarray(ts_us, dtype=np.int64)
    out = np.full(ts.shape, -1, dtype=np.int8)
    out[(ts >= TRAIN_START_US) & (ts < TRAIN_END_US)] = 0
    out[(ts >= TRAIN_END_US) & (ts < VALIDATION_END_US)] = 1
    out[(ts >= VALIDATION_END_US) & (ts < TEST_END_US)] = 2
    return out


def subperiod_of(ts_us: np.ndarray) -> np.ndarray:
    """0 VAL_A [11-16, 11-20), 1 VAL_B [11-20, 11-23), 2 NONE."""
    ts = np.asarray(ts_us, dtype=np.int64)
    out = np.full(ts.shape, 2, dtype=np.int8)
    out[(ts >= TRAIN_END_US) & (ts < VS_END_US)] = 0
    out[(ts >= VS_END_US) & (ts < VALIDATION_END_US)] = 1
    return out


class TestGateError(PermissionError):
    """Raised whenever a TEST row, label, feature or manifest is requested before ALL_METHODS_LOCK."""


def refuse_test_split(split: str) -> None:
    if str(split).upper() in TEST_SPLIT_NAMES or "TEST" in str(split).upper():
        raise TestGateError(
            f"split {split!r} is a TEST split: no RETRO_TEST row, label or feature is read before "
            "ALL_METHODS_LOCK (MANIFEST_SPEC_V4 global_rules.test_gate; EVALUATION_CONTRACT_V4 E10.1)")


# --------------------------------------------------------------------------------------------- hashing
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def canonical_json_bytes(obj) -> bytes:
    """The canonicalization used for every self-hash in V4 (schema 2.1.0 schema_self_sha256_canonicalization)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def self_hash(obj: Mapping, field: str) -> str:
    d = dict(obj)
    d.pop(field, None)
    return sha256_bytes(canonical_json_bytes(d))


def code_tree_sha256(files: Sequence[Path], root: Path) -> str:
    """sha256 over sorted (relative posix name, bytes) pairs; used for builder_code_sha256."""
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.relative_to(root).as_posix()):
        name = p.relative_to(root).as_posix().encode("utf-8")
        data = p.read_bytes()
        h.update(struct.pack("<I", len(name)) + name + struct.pack("<Q", len(data)) + data)
    return h.hexdigest()


def builder_code_sha256() -> str:
    files = [p for p in V4DATA.rglob("*.py") if "__pycache__" not in p.parts]
    return code_tree_sha256(files, V4DATA)


DTYPE_CODE = {"int64": 1, "uint64": 2, "int32": 3, "int8": 4, "bool": 5, "string": 6}


def _string_payload(values: np.ndarray) -> bytes:
    """uint32 LE byte length + UTF-8 bytes for every value, in row order (vectorised through per-value tables)."""
    arr = np.asarray(values, dtype=object)
    if arr.size == 0:
        return b""
    uniq, inv = np.unique(arr.astype(str), return_inverse=True)
    if len(uniq) <= 4096:
        enc = np.empty(len(uniq), dtype=object)
        for i, s in enumerate(uniq):
            bs = s.encode("utf-8")
            enc[i] = struct.pack("<I", len(bs)) + bs
        return b"".join(enc[inv].tolist())
    # high-cardinality strings (client_key_hex): fixed 64-byte ASCII hex, vectorised
    bs = np.char.encode(arr.astype(str), "utf-8")
    widths = np.char.str_len(arr.astype(str))
    if np.all(widths == 64):
        mat = np.empty((arr.size, 68), dtype=np.uint8)
        mat[:, :4] = np.frombuffer(struct.pack("<I", 64), dtype=np.uint8)
        mat[:, 4:] = np.frombuffer(bs.astype("S64").tobytes(), dtype=np.uint8).reshape(arr.size, 64)
        return mat.tobytes()
    return b"".join(struct.pack("<I", len(b)) + b for b in bs.tolist())


class EnumValues:
    """An enum-typed string column held as int codes into a registered vocabulary (hashing needs no strings)."""

    def __init__(self, codes: np.ndarray, vocab: Sequence[str]):
        self.codes = np.asarray(codes, dtype=np.int64)
        self.vocab = tuple(vocab)
        if self.codes.size and (self.codes.min() < 0 or self.codes.max() >= len(self.vocab)):
            raise ValueError("enum code outside its registered vocabulary")

    def __len__(self) -> int:
        return int(self.codes.size)

    def strings(self) -> np.ndarray:
        return np.asarray(self.vocab, dtype=object)[self.codes]


def column_payload(dtype: str, values) -> bytes:
    if dtype == "string" and isinstance(values, EnumValues):
        enc = np.empty(len(values.vocab), dtype=object)
        for i, s in enumerate(values.vocab):
            bs = s.encode("utf-8")
            enc[i] = struct.pack("<I", len(bs)) + bs
        return b"".join(enc[values.codes].tolist()) if len(values) else b""
    if dtype == "string":
        return _string_payload(values)
    if dtype == "bool":
        return np.ascontiguousarray(np.asarray(values, dtype=bool).astype(np.uint8)).tobytes()
    npdt = {"int64": "<i8", "uint64": "<u8", "int32": "<i4", "int8": "<i1"}[dtype]
    return np.ascontiguousarray(np.asarray(values).astype(npdt, copy=False)).tobytes()


def table_content_sha256(columns: Sequence[tuple[str, str, object]]) -> str:
    """MANIFEST_SPEC hashing.table_content_sha256: for each column in declared order, uint32 LE name length, UTF-8
    name, uint8 dtype code, then the payload in row order. Independent of the Parquet writer."""
    h = hashlib.sha256()
    for name, dtype, values in columns:
        nb = name.encode("utf-8")
        h.update(struct.pack("<I", len(nb)) + nb + struct.pack("<B", DTYPE_CODE[dtype]))
        h.update(column_payload(dtype, values))
    return h.hexdigest()


def ordered_ids_sha256(decision_id: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(decision_id, dtype="<i8")).tobytes())


def population_sha256(selection_keys_sorted: np.ndarray) -> str:
    """sha256 over each selected user's 8-byte big-endian selection_key, ascending (CEN/COHORT_PROJECTIONS.json L5)."""
    k = np.asarray(selection_keys_sorted, dtype=np.uint64)
    if k.size > 1 and not np.all(k[1:] > k[:-1]):
        raise ValueError("population_sha256 needs strictly ascending, unique selection keys")
    return sha256_bytes(k.astype(">u8").tobytes())


def user_set_sha256(digests_sorted: np.ndarray) -> str:
    """sha256 over the concatenated 32-byte digests (client_key_hex decoded), ascending."""
    d = np.asarray(digests_sorted, dtype="S32")
    if d.size > 1 and not np.all(d[1:] > d[:-1]):
        raise ValueError("user_set_sha256 needs strictly ascending, unique digests")
    return sha256_bytes(d.tobytes())


def candidate_order_hash(output_class: np.ndarray, product_id: Sequence[str]) -> str:
    """sha256 over output_class 0..K-1: int64 LE class, uint32 LE byte length of product_id, product_id bytes."""
    oc = np.asarray(output_class, dtype=np.int64)
    order = np.argsort(oc, kind="stable")
    if not np.array_equal(oc[order], np.arange(oc.size, dtype=np.int64)):
        raise ValueError("catalogue output_class is not exactly 0..K-1")
    h = hashlib.sha256()
    pids = list(product_id)
    for i in order.tolist():
        b = pids[i].encode("utf-8")
        h.update(struct.pack("<q", int(oc[i])) + struct.pack("<I", len(b)) + b)
    return h.hexdigest()


# -------------------------------------------------------------------------------------------- identity
def client_digest(user_id: str) -> bytes:
    """32-byte salted digest (CLIENT_CONTRACT_V1 C1.2). user_id is the canonical raw STRING identity."""
    return hashlib.sha256(SALT + b"|" + user_id.encode("utf-8")).digest()


def digests_of(user_ids: Iterable[str]) -> np.ndarray:
    """numpy 'S32' array of salted digests, in input order."""
    uids = list(user_ids)
    out = np.empty(len(uids), dtype="S32")
    sha = hashlib.sha256
    pre = SALT + b"|"
    for i, u in enumerate(uids):
        out[i] = sha(pre + u.encode("utf-8")).digest()
    return out


def selection_keys_of(digests: np.ndarray) -> np.ndarray:
    """First 8 digest bytes read big-endian as uint64 (CEN/COHORT_PROJECTIONS.json L4)."""
    d = np.asarray(digests, dtype="S32")
    raw = np.frombuffer(d.tobytes(), dtype=np.uint8).reshape(d.size, 32)[:, :8]
    return np.ascontiguousarray(raw).view(">u8").reshape(-1).astype(np.uint64)


def hex_of(digests: np.ndarray) -> np.ndarray:
    """Lowercase 64-char hex per digest, read from the raw 32-byte buffer (numpy 'S' element access would strip
    trailing zero bytes, so it is never used for identity)."""
    d = np.asarray(digests, dtype="S32")
    s = d.tobytes().hex()
    return np.array([s[i * 64:(i + 1) * 64] for i in range(d.size)], dtype=object)


def digest_rows(digests: np.ndarray) -> list[bytes]:
    """Exact 32-byte digests as Python bytes (full width, from the raw buffer)."""
    d = np.asarray(digests, dtype="S32")
    b = d.tobytes()
    return [b[i * 32:(i + 1) * 32] for i in range(d.size)]


# ------------------------------------------------------------------------------------------ safe paths
class PathEscapeError(PermissionError):
    pass


def safe_path(root: Path, rel: str | os.PathLike) -> Path:
    """Resolve `rel` under `root` or raise PathEscapeError (PKG/12 T44).

    Refuses absolute paths, drive letters, any '..' component, and any result whose fully resolved location
    (symlinks followed) is not inside the resolved root. Containment uses path components (os.path.commonpath),
    so a sibling such as root + '_evil' is refused, unlike a string-prefix test."""
    rel_s = os.fspath(rel)
    p = Path(rel_s)
    # platform-independent: a drive prefix or a backslash is refused on every OS (on POSIX 'C:/x' or 'a\\b' would
    # otherwise be silently accepted as a relative name, and mean something else on Windows)
    if p.is_absolute() or p.drive or rel_s.startswith(("/", "\\")) or "\\" in rel_s \
            or (len(rel_s) >= 2 and rel_s[1] == ":" and rel_s[0].isalpha()):
        raise PathEscapeError(f"absolute or platform-dependent output path refused: {rel_s!r}")
    if any(part == ".." for part in p.parts):
        raise PathEscapeError(f"parent-directory component refused: {rel_s!r}")
    root_r = Path(os.path.realpath(root))
    cand = Path(os.path.realpath(root_r / p))
    try:
        common = os.path.commonpath([str(root_r), str(cand)])
    except ValueError as exc:                                         # different drives on Windows
        raise PathEscapeError(f"path {rel_s!r} leaves the workspace") from exc
    if os.path.normcase(common) != os.path.normcase(str(root_r)):
        raise PathEscapeError(f"path {rel_s!r} resolves outside {root_r}: {cand}")
    return cand


# -------------------------------------------------------------------------------------------- atomic io
def _fsync_dir(d: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(root: Path, rel: str, data: bytes) -> Path:
    """Write `data` to root/rel atomically: temp file in the same directory, fsync, os.replace, fsync(dir).
    On any exception the temporary file is removed and a pre-existing target is left untouched (T46)."""
    target = safe_path(root, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if sha256_file(tmp) != sha256_bytes(data):
            raise IOError(f"temporary file verification failed for {target}")
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def json_bytes_strict(obj, indent: int | None = 1) -> bytes:
    """Strict JSON (allow_nan=False raises on NaN/inf) - serialised BEFORE any file is created (T45)."""
    return (json.dumps(obj, indent=indent, sort_keys=False, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_write_json(root: Path, rel: str, obj, indent: int | None = 1) -> Path:
    return atomic_write_bytes(root, rel, json_bytes_strict(obj, indent))


def write_marker(root: Path, rel: str, verified: Mapping[str, str]) -> Path:
    """Success marker, written last and only after the caller verified every file it names."""
    if not verified:
        raise ValueError("a success marker must name at least one verified file")
    return atomic_write_json(root, rel, {"verified_sha256": dict(verified), "utc": utc_now()})


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------ frozen inputs
def contract_path(name: str) -> Path:
    """contracts/<name> when present (local workspace), else the pinned byte copy in v4data/frozen_inputs/."""
    p = V4 / "contracts" / name
    return p if p.is_file() else FROZEN_INPUTS / name


def load_schema() -> dict:
    """FEATURE_SCHEMA_CLEAN_V2.json, refusing anything but version 2.1.0 with the recorded self-hash."""
    p = contract_path("FEATURE_SCHEMA_CLEAN_V2.json")
    s = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if s.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema version {s.get('schema_version')} != {SCHEMA_VERSION}")
    rec = s.get("schema_self_sha256")
    got = self_hash(s, "schema_self_sha256")
    if rec != SCHEMA_SELF_SHA256 or got != SCHEMA_SELF_SHA256:
        raise ValueError(f"schema self-hash mismatch: recorded {rec}, recomputed {got}, pinned {SCHEMA_SELF_SHA256}")
    return s


def load_manifest_spec() -> dict:
    p = contract_path("MANIFEST_SPEC_V4.json")
    spec = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if spec.get("spec_version") != "1.2.0":
        raise ValueError(f"MANIFEST_SPEC version {spec.get('spec_version')} != 1.2.0")
    return spec


def _reject_constant(x):
    raise ValueError(f"non-strict JSON constant {x!r}")
