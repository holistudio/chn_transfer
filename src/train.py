"""
train.py
========

Training loop for the linear-softmax traditional-character classifier.

Run from the repo root:

    python -m src.train --mode all
    python -m src.train --mode diff --epochs 60
    python -m src.train --mode diff --split-strategy leave_font_out --holdout-font NotoSerifCJK
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DataConfig, build_datasets
from .model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train traditional-character classifier")
    # data / dataset
    p.add_argument("--data-root", default="./data/")
    p.add_argument("--mode", choices=["all", "diff"], default="all",
                   help="train on all traditional chars, or only 'diff'-flagged ones")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--split-strategy", choices=["stratified_random", "leave_font_out"],
                   default="stratified_random")
    p.add_argument("--holdout-font", default="NotoSerifCJK")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # optimisation
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    # io
    p.add_argument("--out", default="checkpoints")
    return p.parse_args()


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

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = DataConfig(
        data_root=args.data_root,
        mode=args.mode,
        img_size=args.img_size,
        channels=args.channels,
        train_ratio=args.train_ratio,
        split_strategy=args.split_strategy,
        holdout_font=args.holdout_font,
        augment=not args.no_augment,
        seed=args.seed,
    )
    bundle = build_datasets(cfg)
    meta = bundle.meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_loader = DataLoader(bundle.train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    test_loader = DataLoader(bundle.test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = build_model(meta).to(device)
    print(f"[train] model: in_dim={model.in_dim}  num_classes={model.num_classes}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_top1 = -1.0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"linear_softmax_{args.mode}.pt")

    for epoch in range(1, args.epochs + 1):
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
        train_acc = correct / max(total, 1)
        ev = evaluate(model, test_loader, device, criterion)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
              f"test_loss {ev['loss']:.4f}  test_top1 {ev['top1']:.3f}  test_top5 {ev['top5']:.3f}")

        if ev["n"] > 0 and ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": meta,           # carries idx2id, char_of, mode, dims
                    "args": vars(args),
                },
                ckpt_path,
            )

    print(f"[train] best test top1 = {best_top1:.3f}  ->  saved to {ckpt_path}")

    # --- demonstrate the requested output: logits + softmax for one test image ---
    if len(bundle.test_ds) > 0:
        model.eval()
        with torch.no_grad():
            x, y = bundle.test_ds[0]
            logits = model(x.unsqueeze(0).to(device))[0]
            probs = torch.softmax(logits, dim=0)
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
