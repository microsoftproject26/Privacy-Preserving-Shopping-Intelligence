"""v4train.cpf_stub — the central-personalized (CPF) mode: a registered interface only, for G3 (TRAIN-01 Build 7).

CPF is the capacity-matched central comparator of PF (EVALUATION_CONTRACT_V4 E8.4 labelling; FL-01 personal.py):
central training of the WINNING family with a private per-user vector p_u of the query dimension added to the
query, q_u = q + p_u, p_u initialized to 0, a separate persistent AdamW for the p rows (weight decay 0, separate clip
1.0), and the penalty (lambda / 2) * sum over the DISTINCT users of a batch of ||p_u||^2 once per step (Fable G0(b)
C7(3)). Its training is NOT implemented here; the interface exists so the mode is registered, refuses the losing
family before anything else, and fails loudly instead of silently running the plain central recipe.
"""
from __future__ import annotations

from typing import Mapping

from .selection import assert_downstream_allowed

REGISTERED_FOR = "G3"
MODE_ID = "CPF_CENTRAL"


class CPFNotImplemented(NotImplementedError):
    pass


class CentralPersonalizedMode:
    """Interface: CentralPersonalizedMode(selection_record, family, *, lam, personal_lr, ...).run(...)"""

    mode_id = MODE_ID
    registered_for = REGISTERED_FOR

    def __init__(self, selection_record: Mapping, family: str, *, lam: float = 1e-4, personal_lr: float = 1e-3,
                 personal_clip: float = 1.0):
        assert_downstream_allowed(selection_record, family, "CPF")   # the losing family is refused first
        self.family = str(family).upper()
        self.lam, self.personal_lr, self.personal_clip = float(lam), float(personal_lr), float(personal_clip)

    def run(self, *args, **kwargs):
        raise CPFNotImplemented("CPF central-personalized training is registered for G3 and not implemented in "
                                "TRAIN-01 (stub interface only)")


MODES = {"MAIN": "v4train.trainer.CentralTrainer.run (6.0 EFE, M01 -> M02)",
         "CALIBRATION": "v4train.calibration.run_calibration (2.0 EFE, M05 -> M06, no weights)",
         "CONTINUATION": "v4train.trainer.CentralTrainer.continue_run (6.0 -> <= 8.0, STILL_IMPROVING + budget)",
         MODE_ID: "v4train.cpf_stub.CentralPersonalizedMode (registered for G3; not implemented)"}

__all__ = ["CentralPersonalizedMode", "CPFNotImplemented", "MODES", "MODE_ID", "REGISTERED_FOR"]
