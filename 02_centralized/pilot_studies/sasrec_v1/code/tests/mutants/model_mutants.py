"""In-memory mutants that deliberately break `SASRecCE` invariants by
monkeypatching a LIVE model instance (never `sasrec_model.py` on disk).

Each mutant is a context manager: apply it, re-run the SAME check the
positive test used, and assert the check now reports a violation. If a test
cannot tell the mutant from the real model, the test was not checking the
property it claimed to check (project rule: every mandatory test needs a
negative control that really fails).
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch


@contextmanager
def force_all_positions_valid(model):
    """Breaks PAD invariance: every column (including PAD) is treated as a
    real event regardless of `lengths`, so PAD-column token ids / side values
    leak into causal attention and the readout. Negative control for the
    Fable #1 PAD-invariance tests."""
    orig = model._layout

    def mutant_layout(self, item_tokens, lengths, attention_mask, position_ids=None):
        B, T, valid, lengths_, ar = orig(item_tokens, lengths, attention_mask, position_ids)
        return B, T, torch.ones_like(valid), lengths_, ar

    model._layout = types.MethodType(mutant_layout, model)
    try:
        yield model
    finally:
        model._layout = orig


@contextmanager
def disable_causal_mask(model):
    """Breaks causality: every query attends every key, past AND future.
    Negative control for the causal-mask / future-leakage tests (05 B11)."""
    orig_causal = model.causal.clone()
    model.causal.fill_(True)
    try:
        yield model
    finally:
        model.causal.copy_(orig_causal)


@contextmanager
def blind_column_zero_readout(model):
    """Breaks 'final query is the last REAL item, not column-1 blindly'
    (05 item 16): always reads hidden position 0 regardless of `lengths`."""
    orig_encode = model._encode

    def mutant_encode(self, item_tokens, lengths, attention_mask, side_kw, *,
                       want_sequence, position_ids=None):
        _, seq = orig_encode(item_tokens, lengths, attention_mask, side_kw,
                              want_sequence=True, position_ids=position_ids)
        blind = seq[:, 0, :]
        return blind, (seq if want_sequence else None)

    model._encode = types.MethodType(mutant_encode, model)
    try:
        yield model
    finally:
        model._encode = orig_encode


@contextmanager
def retie_head_weight(model):
    """Registry v3 negative control: `output_weight()` silently returns a VIEW of the input
    table item_embed.weight[3:3+K] again (the v2 tying that caused common-mode drift)."""
    orig = model.output_weight

    def mutant_output_weight(self):
        return self.item_embed.weight[3:3 + self.K]

    model.output_weight = types.MethodType(mutant_output_weight, model)
    try:
        yield model
    finally:
        model.output_weight = orig


@contextmanager
def recenter_by_row_(model):
    """Negative control for recentering: subtracts a PER-CLASS (non-constant) shift from the
    bias, which is NOT loss-neutral (breaks softmax shift invariance)."""
    orig = model.recenter_output_

    def mutant(self):
        with torch.no_grad():
            self.output_bias.sub_(torch.linspace(0.0, 1.0, self.output_bias.numel(), device=self.output_bias.device))

    model.recenter_output_ = types.MethodType(mutant, model)
    try:
        yield model
    finally:
        model.recenter_output_ = orig


@contextmanager
def untie_head_weight(model):
    """Simulates a broken weight tying: `output_weight()` returns an
    independent COPY instead of a view into `item_embed.weight`. Negative
    control for the tying-by-Parameter-identity test (05 item 17, Fable #3)."""
    orig = model.output_weight
    copy_w = model.item_embed.weight[3:3 + model.K].detach().clone()

    def mutant_output_weight(self):
        return copy_w

    model.output_weight = types.MethodType(mutant_output_weight, model)
    try:
        yield model, copy_w
    finally:
        model.output_weight = orig


def corrupt_pad_row(model, eps: float = 1e-3) -> None:
    """One-shot bug injection (not a context manager): simulates a missing
    `padding_idx` enforcement by nudging the PAD row away from zero. Used as
    the negative control for the '100 updates, PAD row stays zero' test."""
    with torch.no_grad():
        model.item_embed.weight[0] += eps
