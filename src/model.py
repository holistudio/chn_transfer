"""
model.py
========

The baseline linear-softmax classifier, plus a small factory that wires the
input/output dimensions from the dataset metadata.
"""

# from __future__ import annotations

import torch
import torch.nn as nn


class LinearSoftmax(nn.Module):
    """Flatten the image and apply a single linear layer -> per-class logits."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))            # logits, shape (B, num_classes)


def build_model(meta: dict) -> LinearSoftmax:
    """Construct the model straight from `DatasetBundle.meta`."""
    return LinearSoftmax(in_dim=meta["in_dim"], num_classes=meta["num_classes"])
