"""Synthetic memorisation gate (Task T-I) -- Stage B engineering-correctness
check.

This is NOT a claim about real-data quality. It only exercises the REAL
`model.py` (`build_model`/`ContextGRU.forward`) and the REAL `objective.py`
(`exact_ce_loss`) on a purely synthetic, self-consistent task where the
target is a deterministic function of the last context token, so it is
learnable in principle. If the wiring (embeddings, packing, the tied output
head, the loss) were broken, this would fail to memorise 256 examples in a
few hundred full-batch gradient steps; passing is evidence the plumbing
works, nothing more.

Arms:
  * F0_item_only            -- item_tokens only.
  * F4_rich_side_information-- all 6 categorical channels + 19 numeric +
                                15 flags + 12 user-context + 6 masks (widths
                                per FABLE_PLAN.md / model.py's F4 spec).
  * negative control        -- F0-shaped model, but EVERY row's item_tokens
                                is overwritten with the SAME constant token
                                (so the model cannot tell rows apart), while
                                the target_class stays the ORIGINAL
                                deterministic target computed from each row's
                                true (now-hidden) last token. This must fail
                                to beat majority-class guessing, proving the
                                gate has power to fail.

Run on Lobot, GPU 0 only:
    CUDA_VISIBLE_DEVICES=0 <lobot python> code/synthetic_memorisation.py \\
        --out artifacts/SYNTHETIC_MEMORISATION.json --log logs/synthetic_memorisation.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import build_model  # noqa: E402
from objective import exact_ce_loss  # noqa: E402

SEED = 20260922
B = 256
L = 50
K_SYN = 1000
KNOWN_START = 3  # matches rees46_eda.config.KNOWN_START -- reserved PAD/MISSING/OOV rows
MAX_UPDATES = 300
LR = 1e-2

NUMERIC_DIM = 19
FLAGS_DIM = 15
USER_CTX_DIM = 12
USER_CTX_MASK_DIM = 6

# Small synthetic vocabularies for the F4 arm's non-item categorical
# channels, each including the 3 reserved rows (PAD/MISSING/OOV).
AUX_VOCAB = {
    "category_tokens": KNOWN_START + 50,
    "brand_tokens": KNOWN_START + 50,
    "main_category_tokens": KNOWN_START + 10,
    "daypart_tokens": KNOWN_START + 5,
    "weekday_tokens": KNOWN_START + 7,
}


def check_gpu_free() -> None:
    """Abort if another user's process is already using a GPU (task brief:
    always run nvidia-smi first)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    rows = [r for r in out.stdout.strip().splitlines() if r.strip()]
    if rows:
        raise RuntimeError(
            "nvidia-smi shows active GPU compute processes; refusing to start:\n"
            + "\n".join(rows))


def target_fn(raw_last: np.ndarray) -> np.ndarray:
    """target = (last_token * 7 + 13) % K_syn, where `raw_last` is the
    synthetic item id in [0, K_syn) (the item_tokens VALUE minus the
    KNOWN_START offset)."""
    return (raw_last.astype(np.int64) * 7 + 13) % K_SYN


def make_positive_batch(rng: np.random.Generator, group: str) -> dict[str, torch.Tensor]:
    """A self-consistent synthetic batch: `lengths` up to L, item_tokens
    drawn i.i.d. over the K_syn synthetic catalogue (offset by
    KNOWN_START), target_class a deterministic function of the LAST valid
    item token. For F4, every other channel is present at the correct
    width, filled with independent noise (uninformative by construction --
    the model must still route the informative signal through item_tokens
    to solve the task; noise channels only add nuisance capacity).
    """
    lengths = rng.integers(1, L + 1, size=B)
    item_tokens = np.zeros((B, L), dtype=np.int64)
    attention_mask = np.zeros((B, L), dtype=bool)
    for i, n in enumerate(lengths):
        item_tokens[i, :n] = rng.integers(0, K_SYN, size=n) + KNOWN_START
        attention_mask[i, :n] = True
    last_valid_index = lengths - 1
    last_token = item_tokens[np.arange(B), last_valid_index]
    raw_last = last_token - KNOWN_START
    target_class = target_fn(raw_last)

    out = {
        "item_tokens": torch.from_numpy(item_tokens),
        "lengths": torch.from_numpy(lengths.astype(np.int64)),
        "attention_mask": torch.from_numpy(attention_mask),
        "target_class": torch.from_numpy(target_class),
        "loss_mask": torch.ones(B, dtype=torch.bool),
    }

    if group == "F4_rich_side_information":
        for ch, vocab in AUX_VOCAB.items():
            arr = np.zeros((B, L), dtype=np.int64)
            for i, n in enumerate(lengths):
                arr[i, :n] = rng.integers(KNOWN_START, vocab, size=n)
            out[ch] = torch.from_numpy(arr)
        num = np.zeros((B, L, NUMERIC_DIM), dtype=np.float32)
        flg = np.zeros((B, L, FLAGS_DIM), dtype=np.int8)
        for i, n in enumerate(lengths):
            num[i, :n] = rng.normal(size=(n, NUMERIC_DIM)).astype(np.float32)
            flg[i, :n] = rng.integers(0, 2, size=(n, FLAGS_DIM)).astype(np.int8)
        out["event_numeric_features"] = torch.from_numpy(num)
        out["event_quality_flags"] = torch.from_numpy(flg)
        out["user_context"] = torch.from_numpy(
            rng.normal(size=(B, USER_CTX_DIM)).astype(np.float32))
        out["user_context_masks"] = torch.from_numpy(
            rng.integers(0, 2, size=(B, USER_CTX_MASK_DIM)).astype(np.float32))

    return out


def make_negative_control_batch(positive: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Same target_class as the F0 positive batch (the TRUE deterministic
    targets, derived from each row's real last token), but EVERY row's
    item_tokens is overwritten with the SAME constant token everywhere, so
    the model literally cannot distinguish one row from another. Learning
    beyond guessing the majority target class is impossible.
    """
    const_token = KNOWN_START + 1  # any valid, non-reserved synthetic item id
    item_tokens = torch.full((B, L), const_token, dtype=torch.long)
    attention_mask = torch.ones((B, L), dtype=torch.bool)
    lengths = torch.full((B,), L, dtype=torch.long)
    return {
        "item_tokens": item_tokens,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "target_class": positive["target_class"].clone(),
        "loss_mask": torch.ones(B, dtype=torch.bool),
    }


def build_vocab_sizes(group: str) -> dict[str, int]:
    vs = {"item_tokens": KNOWN_START + K_SYN}
    if group == "F4_rich_side_information":
        vs.update(AUX_VOCAB)
    return vs


def train_arm(group: str, batch: dict[str, torch.Tensor], device: torch.device,
               *, max_updates: int = MAX_UPDATES, lr: float = LR, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    vocab_sizes = build_vocab_sizes(group)
    product_idx_of_class = np.arange(KNOWN_START, KNOWN_START + K_SYN)

    model = build_model(
        group, vocab_sizes, product_idx_of_class=product_idx_of_class,
        proj_dropout=0.0, readout_dropout=0.0,  # dropout disabled
    ).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)  # weight decay disabled

    batch = {k: v.to(device) for k, v in batch.items()}
    forward_keys = {k: v for k, v in batch.items()
                    if k not in ("target_class", "loss_mask")}

    history = []
    initial_loss = None
    final_loss = None
    final_top1 = None
    updates_run = 0

    for step in range(max_updates):
        opt.zero_grad(set_to_none=True)
        logits = model(**forward_keys)
        loss_sum, n = exact_ce_loss(logits, batch["target_class"], batch["loss_mask"])
        loss = loss_sum / n
        loss.backward()
        opt.step()
        updates_run += 1

        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == batch["target_class"]).float().mean().item()
        loss_val = float(loss.item())
        if step == 0:
            initial_loss = loss_val
        history.append({"step": step, "loss": loss_val, "top1": top1})

    with torch.no_grad():
        logits = model(**forward_keys)
        loss_sum, n = exact_ce_loss(logits, batch["target_class"], batch["loss_mask"])
        final_loss = float((loss_sum / n).item())
        final_top1 = (logits.argmax(dim=-1) == batch["target_class"]).float().mean().item()

    return {
        "group": group, "initial_loss": initial_loss, "final_loss": final_loss,
        "final_top1": final_top1, "updates_run": updates_run,
        "loss_ratio": final_loss / initial_loss if initial_loss else None,
        "history_first_5": history[:5], "history_last_5": history[-5:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    check_gpu_free()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    result = {
        "seed": SEED, "B": B, "L": L, "K_syn": K_SYN, "max_updates": MAX_UPDATES,
        "optimizer": "Adam", "lr": LR, "dropout": 0.0, "weight_decay": 0.0,
        "device": str(device), "gpu_name": gpu_name, "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "arms": {}, "negative_control": {},
    }

    log_lines = []
    t0 = time.time()

    for group in ("F0_item_only", "F4_rich_side_information"):
        rng = np.random.default_rng(SEED if group == "F0_item_only" else SEED + 1)
        batch = make_positive_batch(rng, group)
        r = train_arm(group, batch, device)
        loss_pass = r["final_loss"] <= 0.05 * r["initial_loss"]
        top1_pass = r["final_top1"] >= 0.95
        r["pass"] = bool(loss_pass and top1_pass)
        r["loss_threshold"] = 0.05 * r["initial_loss"]
        r["top1_threshold"] = 0.95
        result["arms"][group] = r
        line = (f"[{group}] initial_loss={r['initial_loss']:.6f} "
                f"final_loss={r['final_loss']:.6f} (threshold<= {r['loss_threshold']:.6f}) "
                f"final_top1={r['final_top1']:.4f} (threshold>=0.95) "
                f"updates={r['updates_run']} PASS={r['pass']}")
        print(line, flush=True)
        log_lines.append(line)

    # Negative control: reuse the F0 positive batch's TRUE targets, but
    # collapse every row's item_tokens to one constant token.
    rng_neg = np.random.default_rng(SEED)
    positive_f0 = make_positive_batch(rng_neg, "F0_item_only")
    neg_batch = make_negative_control_batch(positive_f0)
    target_np = positive_f0["target_class"].numpy()
    majority_class_rate = float(np.bincount(target_np, minlength=K_SYN).max() / B)

    r_neg = train_arm("F0_item_only", neg_batch, device, seed=SEED + 2)
    threshold = majority_class_rate + 0.05
    neg_pass = r_neg["final_top1"] <= threshold
    result["negative_control"] = {
        "description": ("F0-shaped model trained on the SAME true targets as the F0 "
                         "positive arm, but every row's item_tokens replaced by one "
                         "constant token -- learning beyond majority-class guessing is "
                         "impossible."),
        "majority_class_rate": majority_class_rate,
        "threshold_top1_le": threshold,
        "final_top1": r_neg["final_top1"],
        "final_loss": r_neg["final_loss"],
        "initial_loss": r_neg["initial_loss"],
        "updates_run": r_neg["updates_run"],
        "pass": bool(neg_pass),
    }
    line = (f"[negative_control] majority_class_rate={majority_class_rate:.4f} "
            f"threshold<= {threshold:.4f} final_top1={r_neg['final_top1']:.4f} "
            f"PASS={neg_pass}")
    print(line, flush=True)
    log_lines.append(line)

    result["elapsed_s"] = time.time() - t0
    result["overall_pass"] = bool(
        result["arms"]["F0_item_only"]["pass"]
        and result["arms"]["F4_rich_side_information"]["pass"]
        and result["negative_control"]["pass"]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k not in ("arms",)}, indent=2))
    print(f"wrote {out_path}")
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
