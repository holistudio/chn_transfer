"""
train.py
========

Training loop for the traditional-character classifier.

Run from the repo root:

    python -m src.train --config config.json

All options (data paths, model architecture, optimiser, I/O) come from the
JSON file.  See config.json for the full set of knobs.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DataConfig, build_datasets
from .model import build_model, build_resnet


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def data_config_from_cfg(cfg: dict) -> DataConfig:
    """Translate the 'data' block of config.json into a DataConfig instance."""
    d = cfg["data"]
    return DataConfig(
        data_root=d.get("data_root", "./data/"),
        manifest_dir=d.get("manifest_dir", "./reports/"),
        csv_name=d.get("csv_name", "trad_simpl_clean.csv"),
        img_subdir=d.get("img_subdir", "img"),
        mode=d.get("mode", "all"),
        img_size=d.get("img_size", 128),
        channels=d.get("channels", 1),
        train_ratio=d.get("train_ratio", 0.8),
        split_strategy=d.get("split_strategy", "stratified_random"),
        holdout_font=d.get("holdout_font", "NotoSerifCJK"),
        augment=d.get("augment", True),
        seed=d.get("seed", 42),
    )


def build_model_from_config(cfg: dict, meta: dict) -> nn.Module:
    """
    Dispatch on cfg['model']['type'] to instantiate the right architecture.

    in_dim and num_classes are always derived from the dataset metadata so
    that the model and data stay consistent regardless of what the config says.
    Adding a new architecture: implement it in model.py, add a branch here.
    """
    model_type = cfg["model"]["type"]
    params = cfg["model"].get("params", {})

    if model_type == "linear_softmax":
        return build_model(meta)                    # params unused; no hidden layers

    elif model_type == "mlp":
        raise NotImplementedError(
            "mlp is listed in the schema but not yet implemented in model.py. "
            "Add an MLP class there and wire it up here."
        )

    elif model_type in ("resnet", "cnn"):
        return build_resnet(meta, **params)         # params: layers, width

    else:
        raise ValueError(
            f"Unknown model.type {model_type!r}. "
            "Expected one of: linear_softmax, mlp, resnet"
        )


# --------------------------------------------------------------------------- #
# Evaluation (unchanged from original)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> dict:
    if len(loader.dataset) == 0:
        return {"loss": float("nan"), "top1": float("nan"), "top5": float("nan"), "n": 0}
    model.eval()
    loss_sum = correct1 = correct5 = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        k = min(5, logits.size(1))
        top = logits.topk(k, dim=1).indices
        correct1 += (top[:, 0] == y).sum().item()
        correct5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
    return {"loss": loss_sum / total, "top1": correct1 / total,
            "top5": correct5 / total, "n": total}


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train traditional-character classifier")
    p.add_argument("--config", required=True, help="Path to config.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg  = data_config_from_cfg(cfg)
    train_cfg = cfg.get("training", {})
    io_cfg    = cfg.get("io", {})
    model_type = cfg["model"]["type"]

    torch.manual_seed(data_cfg.seed)
    bundle = build_datasets(data_cfg)
    meta = bundle.meta
    meta["model_type"] = model_type             # stored in checkpoint for predict.py
    meta["model_params"] = cfg["model"].get("params", {})  # rebuild args for predict.py

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  config={args.config}")

    batch_size  = train_cfg.get("batch_size", 128)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(bundle.train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(bundle.test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    model = build_model_from_config(cfg, meta).to(device)
    print(f"[train] model={model_type}  in_dim={meta['in_dim']}  "
          f"num_classes={meta['num_classes']}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    epochs = train_cfg.get("epochs", 40)
    best_top1 = -1.0
    checkpoints_dir = io_cfg.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"{model_type}_{data_cfg.mode}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running / max(total, 1)
        train_acc  = correct / max(total, 1)
        ev = evaluate(model, test_loader, device, criterion)
        print(f"epoch {epoch:3d}/{epochs}  "
              f"loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
              f"test_loss {ev['loss']:.4f}  test_top1 {ev['top1']:.3f}  "
              f"test_top5 {ev['top5']:.3f}")

        if ev["n"] > 0 and ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": meta,       # carries idx2id, char_of, dims, model_type, …
                    "cfg": cfg,         # full config snapshot for reproducibility
                },
                ckpt_path,
            )

    print(f"[train] best test top1 = {best_top1:.3f}  ->  saved to {ckpt_path}")

    # --- demo: one test image -> top predictions ---
    if len(bundle.test_ds) > 0:
        model.eval()
        with torch.no_grad():
            x, y = bundle.test_ds[0]
            logits = model(x.unsqueeze(0).to(device))[0]
            probs  = torch.softmax(logits, dim=0)
            k = min(5, logits.numel())
            top = probs.topk(k)
            idx2id, char_of = meta["idx2id"], meta["char_of"]
            print("\n[demo] one test image, top predictions:")
            print(f"       true trad_id = {idx2id[y]} ({char_of.get(idx2id[y], '?')})")
            for p, i in zip(top.values.tolist(), top.indices.tolist()):
                tid = idx2id[i]
                print(f"       {tid} ({char_of.get(tid, '?')})  "
                      f"logit={logits[i].item():+.3f}  prob={p:.4f}")


if __name__ == "__main__":
    main()