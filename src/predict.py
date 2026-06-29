"""
predict.py
==========

Evaluate a trained checkpoint over the ENTIRE recorded test set and write a JSON
report logging inference performance on every test entry, plus overall and
per-font aggregates.

    python -m src.predict --config config.json

The checkpoint to load is resolved in this order:
  1. predict.checkpoint in config.json  (explicit path)
  2. {io.checkpoints_dir}/{model.type}_{data.mode}.pt  (derived, matches train.py naming)

The report is written to:
  {io.reports_dir}/eval_{model.type}_{data.mode}.json

For quick single-image scoring, use --image to bypass the full test-set eval:

    python -m src.predict --config config.json --image path/to/char.png

This overrides predict.single_image in the config file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from .dataset import DataConfig, build_transforms
from .model import LinearSoftmax, build_resnet


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(cfg: dict) -> str:
    """
    Return the checkpoint path to load.
    Explicit predict.checkpoint wins; otherwise derive from model type + mode,
    matching the filename that train.py writes.
    """
    explicit = cfg.get("predict", {}).get("checkpoint")
    if explicit:
        return explicit
    model_type      = cfg["model"]["type"]
    mode            = cfg["data"]["mode"]
    checkpoints_dir = cfg.get("io", {}).get("checkpoints_dir", "checkpoints")
    return os.path.join(checkpoints_dir, f"{model_type}_{mode}.pt")


def _resolve_report_path(cfg: dict) -> str:
    model_type  = cfg["model"]["type"]
    mode        = cfg["data"]["mode"]
    reports_dir = cfg.get("io", {}).get("reports_dir", "reports")
    return os.path.join(reports_dir, f"eval_{model_type}_{mode}.json")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    # Dispatch on the model type recorded at training time.
    # New architectures: add branches here to match train.py's build_model_from_config.
    model_type = meta.get("model_type", "linear_softmax")   # fallback for old checkpoints
    if model_type == "linear_softmax":
        model = LinearSoftmax(in_dim=meta["in_dim"], num_classes=meta["num_classes"])
    elif model_type in ("resnet", "cnn"):
        model = build_resnet(meta, **meta.get("model_params", {}))
    else:
        raise ValueError(
            f"Checkpoint was trained with model_type={model_type!r} which is not "
            "yet supported by predict.py. Add a branch here to match train.py."
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, meta


def _load_manifest(meta: dict, manifest_path: Optional[str]) -> List[dict]:
    """Prefer an explicit CSV; otherwise use the manifest baked into the checkpoint."""
    if manifest_path:
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["label"] = int(r["label"])
        return rows
    manifest = meta.get("test_manifest")
    if not manifest:
        raise RuntimeError(
            "No test manifest found in checkpoint meta and no predict.manifest given. "
            "Re-train with the manifest-saving dataset, or set predict.manifest in config.json."
        )
    return manifest


# --------------------------------------------------------------------------- #
# Scoring (unchanged from original)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_image(model: nn.Module, meta: dict, image_path: str,
                  device: torch.device, tfm, topk: int = 5
                  ) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """Returns (logits, probs, topk_list) for one image; topk maps back to trad_ids."""
    img = Image.open(image_path).convert("L")
    x = tfm(img).unsqueeze(0).to(device)

    logits = model(x)[0]
    probs  = torch.softmax(logits, dim=0)

    k = min(topk, logits.numel())
    top = probs.topk(k)
    idx2id, char_of = meta["idx2id"], meta["char_of"]
    results = [
        {
            "trad_id": idx2id[i],
            "char": char_of.get(idx2id[i], "?"),
            "logit": logits[i].item(),
            "prob": p,
        }
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    return logits.cpu(), probs.cpu(), results


def _blank_stats() -> dict:
    return {"n": 0, "correct_top1": 0, "correct_top5": 0,
            "sum_true_prob": 0.0, "sum_loss": 0.0}


def _finalize(stats: dict) -> dict:
    n = max(stats["n"], 1)
    return {
        "n": stats["n"],
        "top1_acc": stats["correct_top1"] / n,
        "top5_acc": stats["correct_top5"] / n,
        "mean_true_prob": stats["sum_true_prob"] / n,
        "mean_loss": stats["sum_loss"] / n,
    }


@torch.no_grad()
def evaluate_manifest(model: nn.Module, meta: dict, manifest: List[dict],
                      device: torch.device, topk: int = 5) -> dict:
    """Run every test entry through the model and build a per-entry + aggregate report."""
    cfg_inner = DataConfig(img_size=meta["img_size"], channels=meta["channels"], augment=False)
    tfm = build_transforms(cfg_inner, train=False)
    idx2id, char_of = meta["idx2id"], meta["char_of"]

    entries: List[dict] = []
    overall: dict = _blank_stats()
    per_font: Dict[str, dict] = defaultdict(_blank_stats)

    for row in manifest:
        path, label, font = row["path"], row["label"], row.get("font", "")
        logits, probs, results = predict_image(model, meta, path, device, tfm, topk)

        true_prob  = probs[label].item()
        true_logit = logits[label].item()
        loss       = -math.log(max(true_prob, 1e-12))
        top_idx    = [meta["id2idx"][r["trad_id"]] for r in results]
        correct1   = top_idx[0] == label
        correct5   = label in top_idx[:5]

        for bucket in (overall, per_font[font]):
            bucket["n"] += 1
            bucket["correct_top1"] += int(correct1)
            bucket["correct_top5"] += int(correct5)
            bucket["sum_true_prob"] += true_prob
            bucket["sum_loss"] += loss

        entries.append({
            "path": path,
            "trad_id": row.get("trad_id", idx2id.get(label, "?")),
            "char": row.get("char", char_of.get(idx2id.get(label, ""), "?")),
            "font": font,
            "true_label": label,
            "true_prob": true_prob,
            "true_logit": true_logit,
            "loss": loss,
            "pred_trad_id": results[0]["trad_id"],
            "pred_char": results[0]["char"],
            "correct_top1": correct1,
            "correct_top5": correct5,
            "topk": results,
        })

    return {
        "checkpoint_mode": meta.get("mode"),
        "model_type": meta.get("model_type", "linear_softmax"),
        "num_classes": meta.get("num_classes"),
        "split_strategy": meta.get("split_strategy"),
        "holdout_font": meta.get("holdout_font"),
        "summary": _finalize(overall),
        "per_font": {f: _finalize(s) for f, s in sorted(per_font.items())},
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint over the test set")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--image", default=None,
                    help="Score a single image and print results (overrides predict.single_image in config)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    predict_cfg = cfg.get("predict", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve_checkpoint(cfg)
    model, meta = load_model(ckpt_path, device)
    print(f"[eval] loaded {ckpt_path}  model={meta.get('model_type', 'linear_softmax')}  "
          f"mode={meta.get('mode')}  classes={meta.get('num_classes')}")

    # --image CLI arg takes precedence over predict.single_image in the config
    single_image = args.image or predict_cfg.get("single_image")
    topk = predict_cfg.get("topk", 5)

    # --- single-image convenience mode ---
    if single_image:
        dc  = DataConfig(img_size=meta["img_size"], channels=meta["channels"], augment=False)
        tfm = build_transforms(dc, train=False)
        _logits, _probs, results = predict_image(model, meta, single_image, device, tfm, topk)
        print(f"mode={meta['mode']}  top-{topk} for {single_image}:")
        for r in results:
            print(f"  {r['trad_id']} ({r['char']})  logit={r['logit']:+.3f}  prob={r['prob']:.4f}")
        return

    # --- full test-set evaluation -> JSON ---
    manifest_path = predict_cfg.get("manifest")
    manifest = _load_manifest(meta, manifest_path)
    report   = evaluate_manifest(model, meta, manifest, device, topk)

    report_path = _resolve_report_path(cfg)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    s = report["summary"]
    print(f"[eval] strategy={report['split_strategy']}  n={s['n']}")
    print(f"[eval] top1={s['top1_acc']:.3f}  top5={s['top5_acc']:.3f}  "
          f"mean_true_prob={s['mean_true_prob']:.3f}  mean_loss={s['mean_loss']:.3f}")
    for font, fs in report["per_font"].items():
        print(f"   {font:<14} n={fs['n']:<5} top1={fs['top1_acc']:.3f}  top5={fs['top5_acc']:.3f}")
    print(f"[eval] wrote per-entry report -> {report_path}")


if __name__ == "__main__":
    main()