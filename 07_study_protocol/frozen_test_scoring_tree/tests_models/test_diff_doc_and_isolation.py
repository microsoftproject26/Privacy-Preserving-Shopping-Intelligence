"""Provenance: DIFF_VS_SNAPSHOT.md is the exact live diff (one justification per hunk); snapshot isolation and hashes.

  * every ```diff block of v4models/DIFF_VS_SNAPSHOT.md equals the re-generated unified diff of the copy against its
    snapshot source, in order, with exactly one justification paragraph per hunk;
  * the diff never touches the forward math lines that define the topology (no removed line contains a layer call
    other than the width-sized constructors named in the document), except the SCHEMA-01 class-C lines, which are
    pinned here one by one (any other removed line fails) and are the identity at L0 (test_schema01_derived.py);
  * ppsi_env isolation: every snapshot module is loaded from code/snap_v1 and its bytes match SNAPSHOT_MANIFEST.json;
    the V2 modules are separate files (never the snapshot modules) and no bytecode cache exists under snap_v1.
Negative control: a document with one hunk altered is detected.
"""
from __future__ import annotations

from pathlib import Path

import ppsi_env
from models_testkit import nc
from v4models import diffdoc, gru_v2, sasrec_v2


def test_diff_document_matches_the_live_diff():
    assert diffdoc.check() == []
    st = diffdoc.stats()
    assert st["v4models/gru_v2.py"]["hunks"] == 20 and st["v4models/sasrec_v2.py"]["hunks"] == 16   # + SCHEMA-01R1
    assert st["v4models/gru_v2.py"]["removed"] == 5, "SCHEMA-01R1 changes the GRU copy by pure additions only"


# SCHEMA-01 class-C removed lines (DIFF_VS_SNAPSHOT.md hunks S4-S12): the module-constant side-token list and the
# unconditional rich branches; every one is replaced by the same statement over `self.side_channels` / a branch flag.
SCHEMA01_REMOVED = [
    'self._categorical_channels = ("item_tokens",) + (SIDE_CHANNELS if cfg.rich else ())',
    'self.use_user_context = bool(cfg.rich)',
    'self.side_embeds = nn.ModuleDict({ch: nn.Embedding(int(vocab_sizes[ch]), SIDE_DIMS[ch])',
    'for ch in SIDE_CHANNELS})',
    'side_width = sum(SIDE_DIMS.values()) + NUMFLAG_OUT',
    'self.side_proj = nn.Linear(side_width, d)',
    'self.side_norm = nn.LayerNorm(d, eps=cfg.ln_eps)',
    'self.side_gate = nn.Parameter(torch.tensor(GATE_INIT))',
    'self.user_proj = nn.Linear(USER_HIDDEN, d)',
    'self.user_norm = nn.LayerNorm(d, eps=cfg.ln_eps)',
    'self.user_gate = nn.Parameter(torch.tensor(GATE_INIT))',
    'for ch in SIDE_CHANNELS:',
    'self._xavier_linear(lin, g2)',
    'ln.weight.fill_(1.0); ln.bias.zero_()',
    'self.side_gate.fill_(GATE_INIT); self.user_gate.fill_(GATE_INIT)',
    'parts = [self.side_embeds[ch](tok_sel[ch]) for ch in SIDE_CHANNELS]',
    'nf = torch.cat([num_sel.to(torch.float32), flag_sel.to(torch.float32)], dim=-1)',
    'parts.append(F.gelu(self.numflag_proj(nf)))',
    'need = SIDE_CHANNELS + ("event_numeric_features", "event_quality_flags",',
    '"user_context", "user_context_masks")',
    'rich_on = self.cfg.rich and self.side_enabled',
    'tok_sel = {ch: side_kw[ch][:, :T].reshape(-1).index_select(0, flat_idx) for ch in SIDE_CHANNELS}',
    'num = side_kw["event_numeric_features"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)',
    'flg = side_kw["event_quality_flags"][:, :T].reshape(B * T, -1).index_select(0, flat_idx)',
    'tok_sel = {ch: side_kw[ch][:, :T] for ch in SIDE_CHANNELS}',
    'x = x + self._side_residual(tok_sel, side_kw["event_numeric_features"][:, :T],',
    'side_kw["event_quality_flags"][:, :T])',
    'if self.cfg.rich and self.side_enabled:',
]


def test_removed_lines_are_width_lines_only():
    removed = []
    for snap_rel, v2_rel in diffdoc.PAIRS:
        removed += [ln[1:].strip() for ln in diffdoc.diff_lines(snap_rel, v2_rel)[2:]
                    if ln.startswith("-") and not ln.startswith("---")]
    allowed_tokens = ("_USER_CONTEXT_DIM", "_USER_CONTEXT_MASK_DIM", "NUMERIC_DIM", "FLAGS_DIM", "USER_CTX_DIM",
                      "USER_MASK_DIM", "numeric(19)+flags(15)=34", "# [B, 18]", "impl: str = \"unpadded\") -> None:",
                      "vocab_sizes: Optional[dict] = None) -> SASRecCE:", "seed=seed, impl=impl)")
    bad = [ln for ln in removed if not any(t in ln for t in allowed_tokens)]
    assert sorted(bad) == sorted(SCHEMA01_REMOVED), f"removed lines outside the width / SCHEMA-01 scope: "         f"{sorted(set(bad) - set(SCHEMA01_REMOVED))}; pinned but not removed: {sorted(set(SCHEMA01_REMOVED) - set(bad))}"


def test_snapshot_isolation_hashes_and_no_bytecode():
    seen = ppsi_env.assert_isolated()
    ppsi_env.verify_snapshot()
    snap = ppsi_env.SNAP.resolve()
    for name in ("model", "train", "sasrec_model", "objective", "sasrec_objective"):
        if name in seen:
            assert Path(seen[name]).resolve().is_relative_to(snap)
    for mod in (gru_v2, sasrec_v2):
        assert not Path(mod.__file__).resolve().is_relative_to(snap)
    assert not list(snap.rglob("__pycache__")), "a bytecode cache was written into snap_v1"


@nc("a document whose hunk text was edited no longer matches the live diff")
def test_nc_tampered_diff_document():
    import re
    text = diffdoc.DOC.read_text(encoding="utf-8")
    bad = text.replace("+USER_HIDDEN = 64", "+USER_HIDDEN = 65", 1)
    blocks_bad = [m.group(1).rstrip("\n") for m in re.finditer(r"```diff\n(.*?)\n```", bad, flags=re.S)]
    assert blocks_bad == diffdoc.blocks()
