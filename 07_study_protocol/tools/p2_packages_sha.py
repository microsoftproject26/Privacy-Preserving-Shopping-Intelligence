"""Look up the SHA256 of a microsoft-edge-stable .deb in a Debian `Packages` index, and compare.

    python3 p2_packages_sha.py lookup <Packages[.gz]> <version>            -> prints the repo SHA256 (exit 2 if absent)
    python3 p2_packages_sha.py verify <Packages[.gz]> <version> <deb>      -> prints repo/actual; exit 0 match, 1 mismatch
"""
from __future__ import annotations

import gzip
import hashlib
import sys

PKG = "microsoft-edge-stable"


def _read(path: str) -> str:
    raw = open(path, "rb").read()
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_packages(text: str, package: str = PKG) -> dict[str, str]:
    """version -> SHA256 for every stanza of `package`."""
    out: dict[str, str] = {}
    for stanza in text.replace("\r\n", "\n").split("\n\n"):
        fields = {}
        for line in stanza.split("\n"):
            if ":" in line and not line.startswith((" ", "\t")):
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        if fields.get("Package") == package and "Version" in fields and "SHA256" in fields:
            out[fields["Version"]] = fields["SHA256"].lower()
    return out


def verify(text: str, version: str, deb_sha256: str) -> tuple[bool, str | None]:
    repo = parse_packages(text).get(version)
    return (repo is not None and repo == deb_sha256.lower()), repo


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "lookup":
        repo = parse_packages(_read(argv[2])).get(argv[3])
        if repo is None:
            return 2
        print(repo)
        return 0
    if len(argv) == 5 and argv[1] == "verify":
        h = hashlib.sha256()
        with open(argv[4], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ok, repo = verify(_read(argv[2]), argv[3], h.hexdigest())
        print(f"repo={repo} actual={h.hexdigest()}")
        return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
