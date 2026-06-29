"""
model.py
========

Two architectures for the traditional-character classifier:

  * LinearSoftmax  -- the flatten-and-one-linear-layer baseline.
  * ResNet         -- a from-scratch residual CNN that keeps the 2D structure
                      of the glyph instead of flattening it.

Each architecture has a `build_*` factory that wires its dimensions straight
from `DatasetBundle.meta`, so the model always matches the data.
"""

# from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
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
    """Construct the baseline straight from `DatasetBundle.meta`."""
    return LinearSoftmax(in_dim=meta["in_dim"], num_classes=meta["num_classes"])


# --------------------------------------------------------------------------- #
# ResNet
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """Two 3x3 convs + a residual connection (the ResNet-18/34 building block)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # match dimensions on the skip path when shape changes
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet(nn.Module):
    """
    Compact ResNet for single-channel glyph images.

    Defaults to a ResNet-18-style layout (layers=[2,2,2,2]).  An adaptive
    average pool before the classifier makes it agnostic to `img_size`.
    """

    def __init__(self, in_ch: int, num_classes: int,
                 layers: List[int] = (2, 2, 2, 2), width: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 128 -> 64
        )

        self.cur = width
        self.layer1 = self._make_layer(width,     layers[0], stride=1)   # 64
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)   # 32
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)   # 16
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)   # 8

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(width * 8, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        seq = []
        for s in strides:
            seq.append(BasicBlock(self.cur, out_ch, stride=s))
            self.cur = out_ch
        return nn.Sequential(*seq)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)                       # logits, shape (B, num_classes)


def build_resnet(meta: dict, **params) -> ResNet:
    """
    Construct a ResNet from `DatasetBundle.meta`.

    Recognised params (all optional):
      layers : list[int]  -- blocks per stage, default [2, 2, 2, 2]
      width  : int        -- channels in the first stage, default 64
    """
    layers = tuple(params.get("layers", (2, 2, 2, 2)))
    width  = params.get("width", 64)
    return ResNet(
        in_ch=meta["channels"],
        num_classes=meta["num_classes"],
        layers=layers,
        width=width,
    )