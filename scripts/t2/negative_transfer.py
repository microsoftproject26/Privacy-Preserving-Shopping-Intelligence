"""Did fine-tuning the encoder for T2 damage T1?

The plan called for this measurement after every head and it had not been made. It is the
check the whole on-device argument rests on: if one backbone cannot serve three tasks
without the later heads ruining the earlier ones, then `S2-DS-ST1` (shared versus separate
models) already has its answer, and `S2-DS-08`'s loss weighting has a much harder job than
anyone has budgeted for.

The T2 ladder brackets the question exactly:

* **frozen** - not one encoder weight moved, so negative transfer is structurally impossible
  and T1 must come back unchanged. That rung is therefore also a *control*: if it does not
  reproduce the T1 number to the last decimal, this harness is wrong, not the model.
* **fine-tuned** - the encoder moved to serve T2. Whatever T1 loses is the price of sharing,
  measured instead of assumed.

T1 is scored with `S2-DS-01`'s own evaluator, on its own frozen VALIDATION cache, with the
same slice and the same off-diagonal suppression. Writing a second T1 evaluator here is the
precise defect that has already cost this project three wrong numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

T1_TASK = Path(__file__).resolve().parent.parent / "S2-DS-01_GRU_T1_Model"
sys.path.insert(0, str(T1_TASK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize import selected_config  # noqa: E402
from train import CATEGORIES, DEVICE, SPEC, Split, score  # noqa: E402
from ppsi.models.checkpoint import load_encoder as shared_load_encoder  # noqa: E402
from ppsi.models.session_gru import build_model  # noqa: E402

# Read from S2-DS-01, never restated. A config written out a second time here would let the
# two drift, and a T1 model built with different channels would load a T2 checkpoint into
# the wrong shapes - or worse, into the right shapes with the wrong meaning.
ENCODER, _LEARNING_RATE, _SWEEP = selected_config()
SEED = 13

OUTPUT = Path(__file__).resolve().parent / "output"
BASELINE_SLICE_MACRO = 0.3143
T1_PUBLISHED = 0.3474           # mean over seeds 13/42/2026
SEED_SPREAD = 0.0033


def measure(label: str, checkpoint: Path, validation: Split) -> dict:
    model = build_model(SEED, batch_spec=SPEC, config=ENCODER)
    # These checkpoints predate the candidate-rank channel, so their T3 head is a different
    # shape. T1 scoring never reads it. The shared loader drops exactly the T3 tensors and
    # raises if an encoder tensor fails - which matters more here than anywhere else: a
    # silently randomised encoder would show up as catastrophic "negative transfer" and the
    # conclusion would be the opposite of the truth.
    provenance = shared_load_encoder(model, checkpoint)
    assert not [k for k in provenance["new_parameters"]
                if not k.startswith(("t3", "candidate_", "t2_head", "query_"))], (
        f"{checkpoint.name}: an encoder parameter is missing, not just a head one")
    model = model.to(DEVICE).eval()

    measured = score(model, validation, legacy_clamp=False,
                     popularity=np.zeros(CATEGORIES))
    del model
    return {"model": label, "checkpoint": checkpoint.name,
            "slice_macro_mrr": measured["slice_macro"],
            "slice_micro_mrr": measured["slice_micro"],
            "overall_micro_mrr": measured.get("overall_micro")}


def main() -> None:
    validation = Split.load("validation")
    print(f"  T1 VALIDATION decisions: {len(validation):,}\n")

    wanted = [
        ("S2-DS-01 encoder, T1 only", T1_TASK / "output" / "s2_ds_01_gru_t1_seed13.pt"),
        ("after T2, encoder frozen", OUTPUT / "t2_1.pt"),
        ("after T2, encoder fine-tuned", OUTPUT / "t2_2.pt"),
    ]
    rows = [measure(label, path, validation) for label, path in wanted if path.exists()]
    for row in rows:
        print(f"  {row['model']:<30} slice macro MRR {row['slice_macro_mrr']:.4f}")

    reference = rows[0]["slice_macro_mrr"]
    frozen = next((r for r in rows if "frozen" in r["model"]), None)
    tuned = next((r for r in rows if "fine-tuned" in r["model"]), None)

    result = {"task": "S2-DS-06", "question": "did T2 fine-tuning damage T1?",
              "evaluator": "S2-DS-01 train.score, unchanged",
              "baseline_slice_macro": BASELINE_SLICE_MACRO,
              "t1_published_mean_over_seeds": T1_PUBLISHED,
              "seed_spread": SEED_SPREAD, "models": rows}

    if frozen is not None:
        drift = frozen["slice_macro_mrr"] - reference
        result["control_drift"] = round(drift, 6)
        print(f"\n  control: freezing should change nothing.  drift {drift:+.6f}")
        assert abs(drift) < 1e-6, (
            f"the frozen rung moved T1 by {drift:+.6f}; freezing cannot change the encoder, "
            "so this harness is measuring something other than what it claims")

    if tuned is not None:
        cost = tuned["slice_macro_mrr"] - reference
        result["cost_of_sharing"] = round(cost, 4)
        result["beyond_noise"] = bool(abs(cost) > SEED_SPREAD)
        result["verdict"] = (
            "fine-tuning for T2 measurably costs T1" if cost < -SEED_SPREAD else
            "fine-tuning for T2 measurably helps T1" if cost > SEED_SPREAD else
            "no measurable negative transfer: the change is inside the seed spread")
        print(f"  cost of sharing: {cost:+.4f}   against a seed spread of {SEED_SPREAD}")
        print(f"  -> {result['verdict']}")

    (OUTPUT / "s2_ds_06_negative_transfer.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
