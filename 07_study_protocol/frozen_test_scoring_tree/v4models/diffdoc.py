"""v4models.diffdoc — the exact line diff of each V2 model copy against the byte snapshot, and a checker that
DIFF_VS_SNAPSHOT.md embeds exactly that diff (so the document cannot go stale).

  python -B -m v4models.diffdoc            # print the diff blocks in document order
  python -B -m v4models.diffdoc --check    # exit 1 unless DIFF_VS_SNAPSHOT.md matches the live diff

Diff format: difflib.unified_diff with 2 context lines, file headers `--- snap_v1/<path>` / `+++ v4models/<path>`.
In the document every ```diff block is either a file header (the two header lines) or exactly one hunk.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DOC = Path(__file__).resolve().parent / "DIFF_VS_SNAPSHOT.md"
PAIRS = (("snap_v1/reuse/model.py", "v4models/gru_v2.py"),
         ("snap_v1/sasrec_model.py", "v4models/sasrec_v2.py"))
CONTEXT = 2


def diff_lines(snap_rel: str, v2_rel: str) -> list:
    a = (CODE / snap_rel).read_text(encoding="utf-8").splitlines()
    b = (CODE / v2_rel).read_text(encoding="utf-8").splitlines()
    return list(difflib.unified_diff(a, b, fromfile=snap_rel, tofile=v2_rel, n=CONTEXT, lineterm=""))


def blocks() -> list:
    """[header, hunk, hunk, ..., header, hunk, ...] as newline-joined strings, in PAIRS order."""
    out = []
    for snap_rel, v2_rel in PAIRS:
        lines = diff_lines(snap_rel, v2_rel)
        if not lines:
            continue
        out.append("\n".join(lines[:2]))
        cur = []
        for ln in lines[2:]:
            if ln.startswith("@@") and cur:
                out.append("\n".join(cur))
                cur = []
            cur.append(ln)
        if cur:
            out.append("\n".join(cur))
    return out


def doc_blocks(text: str) -> list:
    return [m.group(1).rstrip("\n") for m in re.finditer(r"```diff\n(.*?)\n```", text, flags=re.S)]


def stats() -> dict:
    res = {}
    for snap_rel, v2_rel in PAIRS:
        lines = diff_lines(snap_rel, v2_rel)[2:]
        res[v2_rel] = {"hunks": sum(1 for ln in lines if ln.startswith("@@")),
                       "added": sum(1 for ln in lines if ln.startswith("+")),
                       "removed": sum(1 for ln in lines if ln.startswith("-"))}
    return res


def check(doc: Path = DOC) -> list:
    """Problems found (empty = the document embeds exactly the live diff, one justification per hunk)."""
    text = doc.read_text(encoding="utf-8")
    live, got = blocks(), doc_blocks(text)
    problems = []
    if live != got:
        problems.append(f"diff blocks differ: live {len(live)} blocks, document {len(got)}")
        for i, (x, y) in enumerate(zip(live, got)):
            if x != y:
                problems.append(f"first differing block #{i}:\nLIVE:\n{x}\nDOC:\n{y}")
                break
    n_hunks = sum(1 for b in live if b.startswith("@@"))
    n_just = len(re.findall(r"^\*\*Justification[^*]*\*\*", text, flags=re.M))
    if n_just != n_hunks:
        problems.append(f"{n_hunks} hunks but {n_just} justification paragraphs")
    return problems


if __name__ == "__main__":
    if "--check" in sys.argv:
        p = check()
        print("\n".join(p) if p else "DIFF_VS_SNAPSHOT.md matches the live diff")
        raise SystemExit(1 if p else 0)
    for blk in blocks():
        print("```diff\n" + blk + "\n```\n")
    print(stats())
