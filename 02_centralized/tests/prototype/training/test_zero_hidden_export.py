from __future__ import annotations

import torch
from torch import nn


class ZeroHiddenExportFixture(nn.Module):
    """Graph-capture fixture for the technically selected ZERO_HIDDEN semantics."""

    def forward(self, sequence_output: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last_index = torch.clamp(lengths - 1, min=0)
        rows = torch.arange(lengths.shape[0], device=lengths.device)
        gathered = sequence_output[rows, last_index]
        return torch.where(lengths.unsqueeze(1) > 0, gathered, torch.zeros_like(gathered))


def test_zero_hidden_path_is_torch_export_graphable():
    module = ZeroHiddenExportFixture()
    sequence_output = torch.randn(3, 4, 8)
    lengths = torch.tensor([4, 2, 0], dtype=torch.int64)
    exported = torch.export.export(module, (sequence_output, lengths))
    result = exported.module()(sequence_output, lengths)
    assert torch.equal(result[2], torch.zeros_like(result[2]))
