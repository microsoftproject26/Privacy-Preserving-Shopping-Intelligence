"""fl.peft — parameter-efficient (LoRA) variants of the warm FL recipe
(PREG2 §5.4 WC1–WC4, contracts/PREG2.md be0561ab; packet P2-04).

LoRA (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen 2022, ICLR, "LoRA: Low-Rank Adaptation of Large Language
Models") on the four Linear layers qkv, out, ff1, ff2 of BOTH CausalBlocks of v4models/sasrec_v2.py (SASREC_D64_B2):
  `LoRALinear(base, r, alpha, gen)`:  W_eff = W + (B A) * alpha / r,   y = x W_eff^T + b
      W [out, in], b [out]   the base weight / bias (theta_pre), FROZEN (requires_grad False; keys base_weight /
                             base_bias)
      A [r, in]              trainable, kaiming_uniform_(a = sqrt(5)) (the loralib / HF-PEFT default), drawn on the CPU
                             from ONE torch.Generator seeded derive_seed(seed, "p2_lora"), consumed block 0 -> 1, and
                             within a block in `targets` order (device-independent init)
      B [out, r]             trainable, zeros  -> W_eff = W at init, so the wrapped model scores exactly as theta_pre
      alpha = r registered   -> scale 1
  The forward uses the effective weight (one F.linear), and `weight` is a PROPERTY returning W_eff: any code that reads
  `layer.weight` directly sees the adapted weight (TO-CHECK of P2-04: CausalBlock.forward_tokens / forward_padded call
  self.qkv(...) / self.out(...) / self.ff1(...) / self.ff2(...) as modules and never read `.weight`; only
  SASRecCE._xavier_linear reads lin.weight, at construction time, before LoRA is applied; the property is kept anyway
  so a direct reader can never get the stale base weight). `bias` is a property returning the base bias.
`apply_lora(module, targets, r=, alpha=, seed=)` replaces the layers in place (refused twice).
`freeze_all_but(adapter, trainable_keys, source=None)` — written BESIDE fl.runtime.freeze_item_tables, which is not
  edited: every parameter that is not a trainable key becomes a FIXED buffer of the adapter's manifest (requires_grad
  False: no gradient, out of the optimizer, the clip norm, the FedProx term, extract_shared / the upload, aggregation
  and the DP clip / noise dimension; BaseAdapter.load_state_ verifies a worker's copy bit-identical at every visit).
  The output
  layer is frozen, so its recentering hooks become no-ops (as freeze_item_tables). `source` (the server's broadcast
  state) is the one-time bootstrap copy of every frozen tensor into a worker. The adapter's `fl_frozen_keys` lists the
  frozen parameters (FLRun's server / worker consistency check and fl.comm.visit_bytes(frozen_keys=...) use it).
`merged_state(module)` -> a state dict of the BASE SASREC_D64_B2 architecture (same keys / order as the unwrapped
  model): W_merged = W + B A alpha / r (exactly W where that product is 0, so B = 0 merges to theta_pre bit for bit).
  reeval / export load it into an ordinary model (same graph, same ONNX export).
Checkpoints (`install_peft(run)`): best.pt / endpoint_6.0.pt (FLRun.weights_payload) carry the merged base-
  architecture theta under "theta" plus "p2_lora" (format, meta, the A / B tensors, digests). latest.pt (state_dict)
  carries the LoRA broadcast state itself (A / B shared + the frozen base as FIXED buffers), so FLRun.resume restores
  A / B bitwise through the unchanged runtime; `load_lora_from_payload` restores them from a weights payload.
`lora_init_sha256(adapter, theta_pre_digest)`: the identity rule stays "theta_pre only": B = 0 is checked and the merged
  state must hash to theta_pre's digest; it returns the digest of the LoRA broadcast state (the Server init hash).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapter import BufferRule, ParamManifest, Role, build_manifest
from .numerics import derive_seed, state_digest

LORA_TARGETS = ("qkv", "out", "ff1", "ff2")
LORA_A, LORA_B = "lora_A", "lora_B"
BASE_W, BASE_B = "base_weight", "base_bias"
RECENTER_PAIR = ("output_embed", "output_bias")
P2_LORA_FORMAT = "PPSI_V4_P2_LORA_1"
KAIMING_A = math.sqrt(5)


class PeftError(RuntimeError):
    pass


def _noop_hook() -> None:
    return None


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable rank-r update (see the module docstring)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, gen: torch.Generator):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise PeftError(f"LoRALinear wraps an nn.Linear, not {type(base).__name__}")
        r = int(r)
        if r < 1 or r > min(base.in_features, base.out_features):
            raise PeftError(f"rank {r} outside [1, min(in, out) = {min(base.in_features, base.out_features)}]")
        if base.weight.dtype != torch.float32:
            raise PeftError(f"base weight is {base.weight.dtype}; strict FP32 requires float32")
        self.in_features, self.out_features = int(base.in_features), int(base.out_features)
        self.r, self.alpha = r, float(alpha)
        self.scale = self.alpha / self.r
        dev = base.weight.device
        self.base_weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.base_bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("base_bias", None)
        a = torch.empty(r, self.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=KAIMING_A, generator=gen)
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32, device=dev))

    def delta_weight(self) -> Tensor:
        return torch.matmul(self.lora_B, self.lora_A) * self.scale

    @property
    def weight(self) -> Tensor:                          # the effective weight (TO-CHECK of P2-04)
        return self.base_weight + self.delta_weight()

    @property
    def bias(self) -> Optional[Tensor]:
        return self.base_bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.base_bias)

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        """W + B A scale, and exactly W where B A scale == 0 (no signed-zero flip: B = 0 merges to W bit for bit)."""
        d = self.delta_weight()
        return torch.where(d == 0, self.base_weight, self.base_weight + d)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def lora_modules(module: nn.Module) -> "OrderedDict[str, LoRALinear]":
    return OrderedDict((n, m) for n, m in module.named_modules() if isinstance(m, LoRALinear))


def lora_keys(module: nn.Module) -> tuple:
    """The trainable LoRA parameter names (A, B per wrapped layer) in named_parameters order."""
    mods = lora_modules(module)
    return tuple(n for n, _ in module.named_parameters()
                 if n.rpartition(".")[0] in mods and n.rpartition(".")[2] in (LORA_A, LORA_B))


def lora_numel_formula(d: int, ff_mult: int, n_blocks: int, r: int, targets=LORA_TARGETS) -> int:
    """r (in + out) per wrapped layer: qkv d -> 3d, out d -> d, ff1 d -> ff d, ff2 ff d -> d."""
    dims = {"qkv": (d, 3 * d), "out": (d, d), "ff1": (d, ff_mult * d), "ff2": (ff_mult * d, d)}
    return int(n_blocks) * sum(int(r) * (dims[t][0] + dims[t][1]) for t in targets)


def apply_lora(module: nn.Module, targets: Sequence[str] = LORA_TARGETS, *, r: int, alpha: float, seed: int) -> dict:
    """Wrap `targets` of every CausalBlock in `module.blocks` (SASRecCE: both blocks) with LoRALinear, in place.
    Refused when LoRA is already applied, when the module has no CausalBlocks, or a target is not an nn.Linear.
    Returns the meta record (also stored as module.fl_lora). The adapter built on `module` is stale until
    freeze_all_but(adapter, lora_keys(module)) rebuilds its manifest."""
    targets = tuple(str(t) for t in targets)
    if not targets or len(set(targets)) != len(targets) or any(t not in LORA_TARGETS for t in targets):
        raise PeftError(f"targets must be a non-empty duplicate-free subset of {LORA_TARGETS}, got {targets}")
    if getattr(module, "fl_lora", None) is not None or lora_modules(module):
        raise PeftError("LoRA is already applied to this module")
    blocks = getattr(module, "blocks", None)
    if blocks is None or len(blocks) == 0 or any(type(b).__name__ != "CausalBlock" for b in blocks):
        raise PeftError("apply_lora needs module.blocks = the SASRec CausalBlocks (v4models.sasrec_v2)")
    for i, blk in enumerate(blocks):
        for t in targets:
            if not isinstance(getattr(blk, t, None), nn.Linear):
                raise PeftError(f"blocks.{i}.{t} is not an nn.Linear")
    gen = torch.Generator().manual_seed(derive_seed(int(seed), "p2_lora"))
    for blk in blocks:                                   # fixed consumption order: block, then target
        for t in targets:
            setattr(blk, t, LoRALinear(getattr(blk, t), r, alpha, gen))   # same slot: state_dict order kept
    keys = lora_keys(module)
    named = dict(module.named_parameters())
    meta = {"format": P2_LORA_FORMAT, "r": int(r), "alpha": float(alpha), "scale": float(alpha) / int(r),
            "targets": list(targets), "seed": int(seed), "n_blocks": len(blocks), "lora_keys": list(keys),
            "lora_numel": int(sum(named[k].numel() for k in keys)),
            "init": "A ~ kaiming_uniform_(a=sqrt(5)) from derive_seed(seed, 'p2_lora') on the CPU; B = 0"}
    module.fl_lora = meta
    return dict(meta)


def freeze_all_but(adapter, trainable_keys: Sequence[str], source: Optional[Mapping[str, Tensor]] = None) -> dict:
    """Make `trainable_keys` the ONLY shared (uploaded / aggregated / clipped) parameters of `adapter`; every other
    parameter becomes a FIXED buffer (see the module docstring). Idempotent for the same keys; refused for another
    key set, an unknown key, or an adapter already frozen by fl.runtime.freeze_item_tables (this freeze subsumes it).
    `source` = the server's broadcast state to copy every frozen tensor from (worker bootstrap)."""
    keys = tuple(str(k) for k in trainable_keys)
    if not keys or len(set(keys)) != len(keys):
        raise PeftError(f"trainable keys must be a non-empty set, got {keys}")
    have = getattr(adapter, "fl_trainable_keys", None)
    if have is not None:
        if tuple(have) != keys:
            raise PeftError(f"adapter already restricted to {tuple(have)[:4]}..., not {keys[:4]}...")
        return _freeze_record(adapter)
    if tuple(getattr(adapter, "fl_frozen_keys", ()) or ()):
        raise PeftError("adapter already frozen by freeze_item_tables; freeze_all_but subsumes it (do not combine)")
    module = adapter.module
    named = dict(module.named_parameters())
    unknown = [k for k in keys if k not in named]
    if unknown:
        raise PeftError(f"trainable keys are not parameters of the module: {unknown[:4]}")
    frozen = tuple(k for k in named if k not in keys)
    if source is not None:                               # the one-time bootstrap download of the frozen tensors
        missing = [k for k in frozen if k not in source]
        if missing:
            raise PeftError(f"source lacks frozen tensors {missing[:4]}")
        with torch.no_grad():
            for k in frozen:
                named[k].copy_(source[k])
    for k, p in named.items():
        p.requires_grad_(k in keys)
    old = adapter.manifest
    sd = module.state_dict(keep_vars=True)
    rules = {k: e.rule for k, e in old.entries.items()
             if e.role == Role.BUFFER and k in sd and k not in named and e.rule is not None}
    new = build_manifest(module, buffer_rules=rules)
    if set(new.shared_keys) != set(keys):
        raise PeftError(f"manifest shared keys {new.shared_keys[:4]}... != the trainable keys")
    adapter.manifest = ParamManifest(new.entries, new.nonpersistent_buffers, old.private_extra)
    adapter._shared = [named[k] for k in adapter.manifest.shared_keys]
    adapter._buf = {k: sd[k] for k in adapter.manifest.buffer_keys}
    fz = tuple(k for k in adapter.manifest.buffer_keys if k in named)
    if any(k in fz for k in RECENTER_PAIR):
        if not all(k in fz for k in RECENTER_PAIR):
            raise PeftError(f"the output layer {RECENTER_PAIR} is frozen together or not at all")
        adapter.post_step = _noop_hook                   # frozen output layer: no recentering (bits kept)
        adapter.server_post_aggregate = _noop_hook
        adapter.has_post_step = False
    adapter.fl_frozen_keys = fz
    adapter.fl_trainable_keys = keys
    return _freeze_record(adapter)


def _freeze_record(adapter) -> dict:
    m = adapter.manifest
    fz = tuple(getattr(adapter, "fl_frozen_keys", ()) or ())
    return {"trainable_keys": list(m.shared_keys), "shared_numel": m.shared_numel, "shared_bytes": m.shared_bytes,
            "frozen_keys_n": len(fz), "frozen_numel": sum(m.entries[k].numel for k in fz),
            "frozen_bytes": sum(m.entries[k].nbytes for k in fz), "manifest_digest": m.digest()}


# ------------------------------------------------------------------------------------------------ merge / state
@torch.no_grad()
def merged_state(module: nn.Module) -> "OrderedDict[str, Tensor]":
    """The BASE-architecture state dict (keys / order of the unwrapped model; see the module docstring)."""
    mods = lora_modules(module)
    out = OrderedDict()
    for k, v in module.state_dict().items():
        prefix, _, leaf = k.rpartition(".")
        m = mods.get(prefix)
        if m is None:
            out[k] = v.detach().clone()
        elif leaf == BASE_W:
            out[f"{prefix}.weight"] = m.merged_weight().detach().clone()
        elif leaf == BASE_B:
            out[f"{prefix}.bias"] = v.detach().clone()
        elif leaf not in (LORA_A, LORA_B):
            raise PeftError(f"unexpected key {k} inside a LoRA layer")
    return out


def lora_state(module: nn.Module) -> "OrderedDict[str, Tensor]":
    named = dict(module.named_parameters())
    return OrderedDict((k, named[k].detach().clone()) for k in lora_keys(module))


@torch.no_grad()
def load_lora_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Restore A / B bitwise (exact key set, dtype and shape checked)."""
    keys = lora_keys(module)
    if set(state) != set(keys):
        raise PeftError(f"LoRA state keys differ: {sorted(set(state) ^ set(keys))[:4]}")
    named = dict(module.named_parameters())
    for k in keys:
        t = state[k]
        if t.dtype != named[k].dtype or tuple(t.shape) != tuple(named[k].shape):
            raise PeftError(f"LoRA {k}: {t.dtype} {tuple(t.shape)} != {named[k].dtype} {tuple(named[k].shape)}")
        named[k].copy_(t.to(named[k].device))


def lora_init_sha256(adapter, theta_pre_digest: str) -> str:
    """The Server init hash of a freshly LoRA-wrapped theta_pre: B = 0 everywhere, merged state == theta_pre (by its
    registered digest); returns state_digest of the LoRA broadcast state."""
    mods = lora_modules(adapter.module)
    if not mods:
        raise PeftError("no LoRA layer on this adapter")
    if any(bool(m.lora_B.detach().ne(0).any()) for m in mods.values()):
        raise PeftError("LoRA init requires B = 0")
    got = state_digest(merged_state(adapter.module))
    if got != theta_pre_digest:
        raise PeftError(f"merged LoRA init {got[:16]}... != theta_pre {str(theta_pre_digest)[:16]}...")
    init = state_digest(adapter.broadcast_state())
    adapter.fl_lora_lineage = {"theta_pre_digest": str(theta_pre_digest), "lora_init_sha256": init}
    return init


def lora_weights_payload(payload: Mapping, adapter) -> dict:
    """A FLRun weights payload with "theta" = the merged base-architecture state and "p2_lora" = the LoRA record."""
    module = adapter.module
    meta = getattr(module, "fl_lora", None)
    if meta is None:
        raise PeftError("no LoRA meta on the module (apply_lora first)")
    out = dict(payload)
    out["theta"] = merged_state(module)
    out["p2_lora"] = {"format": P2_LORA_FORMAT, "meta": dict(meta), "state": dict(lora_state(module)),
                      "lora_theta_digest": state_digest(adapter.broadcast_state()),
                      "merged_theta_digest": state_digest(out["theta"]),
                      "lineage": dict(getattr(adapter, "fl_lora_lineage", None) or {})}   # theta_pre + LoRA-init hashes
    return out


def load_lora_from_payload(adapter, payload: Mapping) -> None:
    """Restore A / B from a weights payload written through install_peft; the frozen base must be the one the payload
    was trained from (checked: the merged digest after loading equals the payload's)."""
    rec = payload.get("p2_lora")
    if not isinstance(rec, Mapping) or rec.get("format") != P2_LORA_FORMAT:
        raise PeftError("payload has no p2_lora record")
    meta = getattr(adapter.module, "fl_lora", None)
    keys_meta = ("r", "alpha", "targets", "n_blocks", "lora_keys")
    if meta is None or any(meta[k] != rec["meta"][k] for k in keys_meta):
        raise PeftError("the adapter's LoRA structure differs from the payload's")
    load_lora_state(adapter.module, rec["state"])
    if state_digest(merged_state(adapter.module)) != rec["merged_theta_digest"]:
        raise PeftError("merged state after loading differs from the payload's (different frozen base?)")


def install_peft(run) -> dict:
    """Wire an fl.runtime.FLRun whose server / workers were prepared with apply_lora + freeze_all_but: best.pt /
    endpoint_6.0.pt then carry the merged base theta + p2_lora. Call after construction AND after resume (idempotent).
    The process pool is refused (frozen tensors live in-process; FLRun refuses it too)."""
    if getattr(run, "pool", None) is not None:
        raise PeftError("LoRA runs use the in-process driver only")
    srv = run.server.adapter
    meta = getattr(srv.module, "fl_lora", None)
    if meta is None or tuple(getattr(srv, "fl_trainable_keys", ()) or ()) != lora_keys(srv.module):
        raise PeftError("the server adapter is not LoRA-wrapped and restricted (apply_lora + freeze_all_but)")
    for w in (run.workers or ()):
        wm = getattr(w.module, "fl_lora", None)
        if wm != meta or tuple(getattr(w, "fl_trainable_keys", ()) or ()) != tuple(srv.fl_trainable_keys):
            raise PeftError("a worker's LoRA structure / trainable keys differ from the server's")
        if w.manifest.digest() != srv.manifest.digest():
            raise PeftError("a worker's manifest differs from the server's")
    if not getattr(run, "_p2_lora_installed", False):
        orig = run.weights_payload

        def weights_payload() -> dict:
            return lora_weights_payload(orig(), srv)

        run.weights_payload = weights_payload
        run._p2_lora_installed = True
    return {"meta": dict(meta), **_freeze_record(srv)}
