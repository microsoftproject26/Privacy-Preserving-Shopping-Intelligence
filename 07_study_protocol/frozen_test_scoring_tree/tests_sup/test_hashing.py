from __future__ import annotations

import hashlib

from v4sup.hashing import sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    data = b"the quick brown fox" * 10000  # exercise the chunked read loop
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_sha256_file_empty(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()
