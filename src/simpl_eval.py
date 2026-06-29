"""
simpl_eval.py
=============

Evaluate a trained checkpoint on the *simplified*-character probe set: for every
trad_id flagged `['diff']` (simplified glyph differs from traditional, and isn't
tangled up in a many-to-one merge), feed the model the simplified rendering and
check whether it still recognises the traditional class. Writes a JSON report
logging inference performance on every probe entry, plus overall and per-font
aggregates.

    python -m src.simpl_eval --config config.json

The checkpoint to load is resolved the same way as predict.py:
  1. predict.checkpoint in config.json  (explicit path)
  2. {io.checkpoints_dir}/{model.type}_{data.mode}.pt  (derived, matches train.py naming)

The report is written to:
  {io.reports_dir}/eval_simpl_{model.type}_{data.mode}.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from .dataset import build_simpl_eval_dataset
from .predict import load_config, _resolve_checkpoint, load_model, predict_image
from .train import data_config_from_cfg


def _resolve_report_path(cfg: dict) -> str:
    model_type  = cfg["model"]["type"]
    mode        = cfg["data"]["mode"]
    reports_dir = cfg.get("io", {}).get("reports_dir", "reports")
    return os.path.join(reports_dir, f"eval_simpl_{model_type}_{mode}.json")


def _blank_stats() -> dict:
    return {"n": 0, "correct_top1": 0, "correct_top10": 0,
            "sum_true_prob": 0.0, "sum_loss": 0.0}


def _finalize(stats: dict) -> dict:
    n = max(stats["n"], 1)
    return {
        "n": stats["n"],
        "top1_acc": stats["correct_top1"] / n,
        "top10_acc": stats["correct_top10"] / n,
        "mean_true_prob": stats["sum_true_prob"] / n,
        "mean_loss": stats["sum_loss"] / n,
    }


@torch.no_grad()
def evaluate_simpl_samples(model: nn.Module, meta: dict, samples: List[tuple],
                           simpl_char_of: Dict[str, str],
                           device: torch.device, tfm) -> dict:
    """Run every (path, label, font) probe entry through the model."""
    idx2id = meta["idx2id"]

    entries: List[dict] = []
    overall: dict = _blank_stats()
    per_font: Dict[str, dict] = defaultdict(_blank_stats)

    for path, label, font in samples:
        logits, probs, results = predict_image(model, meta, path, device, tfm, topk=10)

        true_prob  = probs[label].item()
        true_logit = logits[label].item()
        loss       = -math.log(max(true_prob, 1e-12))
        top_idx    = [meta["id2idx"][r["trad_id"]] for r in results]
        correct1   = top_idx[0] == label
        correct10  = label in top_idx

        for bucket in (overall, per_font[font]):
            bucket["n"] += 1
            bucket["correct_top1"] += int(correct1)
            bucket["correct_top10"] += int(correct10)
            bucket["sum_true_prob"] += true_prob
            bucket["sum_loss"] += loss

        trad_id = idx2id.get(label, "?")
        entries.append({
            "path": path,
            "trad_id": trad_id,
            "char": simpl_char_of.get(trad_id, "?"),
            "font": font,
            "true_label": label,
            "true_prob": true_prob,
            "true_logit": true_logit,
            "loss": loss,
            "top1": results[0],
            "top10": results,
            "correct_top1": correct1,
            "correct_top10": correct10,
        })

    return {
        "checkpoint_mode": meta.get("mode"),
        "model_type": meta.get("model_type", "linear_softmax"),
        "num_classes": meta.get("num_classes"),
        "summary": _finalize(overall),
        "per_font": {f: _finalize(s) for f, s in sorted(per_font.items())},
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the simplified-character probe set"
    )
    ap.add_argument("--config", required=True, help="Path to config.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve_checkpoint(cfg)
    model, meta = load_model(ckpt_path, device)
    print(f"[simpl_eval] loaded {ckpt_path}  model={meta.get('model_type', 'linear_softmax')}  "
          f"mode={meta.get('mode')}  classes={meta.get('num_classes')}")

    data_cfg = data_config_from_cfg(cfg)
    data_cfg.img_size = meta["img_size"]
    data_cfg.channels = meta["channels"]
    data_cfg.augment = False

    ds, ds_meta = build_simpl_eval_dataset(data_cfg, meta["id2idx"])
    tfm = ds.transform

    report = evaluate_simpl_samples(model, meta, ds.samples, ds_meta["simpl_char_of"],
                                    device, tfm)
    report["dropped_no_image"] = ds_meta["dropped_no_image"]
    report["dropped_not_in_label_space"] = ds_meta["dropped_not_in_label_space"]

    report_path = _resolve_report_path(cfg)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    s = report["summary"]
    print(f"[simpl_eval] n={s['n']}")
    print(f"[simpl_eval] top1={s['top1_acc']:.3f}  top10={s['top10_acc']:.3f}  "
          f"mean_true_prob={s['mean_true_prob']:.3f}  mean_loss={s['mean_loss']:.3f}")
    for font, fs in report["per_font"].items():
        print(f"   {font:<14} n={fs['n']:<5} top1={fs['top1_acc']:.3f}  top10={fs['top10_acc']:.3f}")
    print(f"[simpl_eval] wrote per-entry report -> {report_path}")


if __name__ == "__main__":
    main()
