"""
dataset.py
==========

Loads the rendered *traditional* Chinese character images and turns them into a
train / test split suitable for closed-set classification, where every class is
a `trad_id` (e.g. "A04679").
"""

from __future__ import annotations

import ast
import csv
import glob
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# A sample is (image_path, label_index, font_name).
Sample = Tuple[str, int, str]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    data_root: str = "./data/"                     # contains the csv and img/
    manifest_dir: str = "./reports/"               # contains the test manifest
    csv_name: str = "trad_simpl_clean.csv"
    img_subdir: str = "img"

    mode: str = "all"                           # "all" or "diff"
    img_size: int = 128
    channels: int = 1                           # glyphs are black-on-white -> grayscale is plenty

    train_ratio: float = 0.75                   # used by the stratified strategy
    split_strategy: str = "stratified_random"   # or "leave_font_out"
    holdout_font: str = "NotoSerifCJK"          # only used by leave_font_out

    augment: bool = True                        # train-time font-jitter augmentation
    seed: int = 1337

    # filled in / overridable; kept here so callers can see derived paths
    extra: dict = field(default_factory=dict)

    @property
    def csv_path(self) -> str:
        return os.path.join(self.data_root, 'clean', self.csv_name)

    @property
    def img_root(self) -> str:
        return os.path.join(self.data_root, self.img_subdir)


# --------------------------------------------------------------------------- #
# CSV parsing / class selection  (pure stdlib, easy to unit test)
# --------------------------------------------------------------------------- #
def _parse_flags(raw: str) -> List[str]:
    """The `flags` column stores a python-list literal, e.g. "['diff']"."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        return list(val) if isinstance(val, (list, tuple)) else [str(val)]
    except (ValueError, SyntaxError):
        return []


def select_diff_only_trad_ids(csv_path: str) -> List[Tuple[str, str, str]]:
    """
    Return an *ordered* list of (trad_id, trad_char, simpl_char) for rows whose
    `flags` column is exactly `['diff']` (no other flags, e.g. many-to-one).
    """
    ids: List[Tuple[str, str, str]] = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["trad_id"].strip()
            if not tid or tid in seen:
                continue
            if _parse_flags(row.get("flags", "")) != ["diff"]:
                continue
            seen.add(tid)
            ids.append((tid, row.get("trad_char", "").strip(), row.get("simpl_char", "").strip()))
    return ids


def select_trad_ids(csv_path: str, mode: str) -> List[Tuple[str, str]]:
    """
    Return an *ordered* list of (trad_id, trad_char) for the chosen mode.

    Order follows the CSV (which is frequency-sorted), so the class<->index
    mapping is deterministic for a given mode without needing a seed.
    """
    if mode not in ("all", "diff"):
        raise ValueError(f"mode must be 'all' or 'diff', got {mode!r}")

    ids: List[Tuple[str, str]] = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["trad_id"].strip()
            if not tid or tid in seen:
                continue
            if mode == "diff" and "diff" not in _parse_flags(row.get("flags", "")):
                continue
            seen.add(tid)
            ids.append((tid, row.get("trad_char", "").strip()))
    return ids


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def _font_of(filename: str, trad_id: str, kind: str = "trad") -> str:
    """trad_A04679_NotoSerifCJK.png -> 'NotoSerifCJK'."""
    base = os.path.splitext(os.path.basename(filename))[0]   # trad_A04679_NotoSerifCJK
    prefix = f"{kind}_{trad_id}_"
    return base[len(prefix):] if base.startswith(prefix) else base


def collect_images(img_root: str, trad_ids: List[Tuple[str, str]]
                   ) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    """
    For each requested trad_id, glob its `trad_*.png` files.

    Returns
    -------
    images_by_id : {trad_id: [(path, font), ...]}  only ids that have >=1 image
    dropped      : list of trad_ids that had no traditional image on disk
    """
    images_by_id: Dict[str, List[Tuple[str, str]]] = {}
    dropped: List[str] = []
    for tid, _char in trad_ids:
        pattern = os.path.join(img_root, tid, f"trad_{tid}_*.png")
        paths = sorted(glob.glob(pattern))
        if not paths:
            dropped.append(tid)
            continue
        images_by_id[tid] = [(p, _font_of(p, tid)) for p in paths]
    return images_by_id, dropped


def collect_simpl_images(img_root: str, trad_ids: List[Tuple[str, str, str]]
                         ) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    """
    For each requested trad_id, glob its `simpl_*.png` files (the simplified
    rendering that lives alongside the traditional one in the same subfolder).

    Returns
    -------
    images_by_id : {trad_id: [(path, font), ...]}  only ids that have >=1 image
    dropped      : list of trad_ids that had no simplified image on disk
    """
    images_by_id: Dict[str, List[Tuple[str, str]]] = {}
    dropped: List[str] = []
    for tid, _trad_char, _simpl_char in trad_ids:
        pattern = os.path.join(img_root, tid, f"simpl_{tid}_*.png")
        paths = sorted(glob.glob(pattern))
        if not paths:
            dropped.append(tid)
            continue
        images_by_id[tid] = [(p, _font_of(p, tid, kind="simpl")) for p in paths]
    return images_by_id, dropped


# --------------------------------------------------------------------------- #
# Splitting strategies
# --------------------------------------------------------------------------- #
def stratified_split(images_by_id: Dict[str, List[Tuple[str, str]]],
                     id2idx: Dict[str, int],
                     train_ratio: float,
                     seed: int) -> Tuple[List[Sample], List[Sample]]:
    """
    Per-class random split across fonts.

    Guarantees: every class keeps >= 1 training image; if a class has >= 2
    images, at least one is held out for test.  Because the split is *within*
    each character, the test images are unseen font renderings of characters the
    model has trained on -> a clean cross-font generalisation signal.
    """
    rng = random.Random(seed)
    train: List[Sample] = []
    test: List[Sample] = []

    for tid, imgs in images_by_id.items():
        label = id2idx[tid]
        imgs = list(imgs)
        rng.shuffle(imgs)                       # randomise font ordering per class
        n = len(imgs)

        n_train = max(1, round(n * train_ratio))
        if n > 1:
            n_train = min(n_train, n - 1)       # keep at least one test image

        for p, font in imgs[:n_train]:
            train.append((p, label, font))
        for p, font in imgs[n_train:]:
            test.append((p, label, font))

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def leave_font_out_split(images_by_id: Dict[str, List[Tuple[str, str]]],
                         id2idx: Dict[str, int],
                         holdout_font: str,
                         seed: int) -> Tuple[List[Sample], List[Sample]]:
    """
    Hold out one entire font for testing -> the strictest cross-font test.
    A class whose *only* rendering is the held-out font is kept in train (it
    cannot be both trained on and tested) so the label space stays complete.
    """
    rng = random.Random(seed)
    train: List[Sample] = []
    test: List[Sample] = []

    for tid, imgs in images_by_id.items():
        label = id2idx[tid]
        held = [(p, f) for p, f in imgs if f == holdout_font]
        rest = [(p, f) for p, f in imgs if f != holdout_font]

        if not rest:                            # only the holdout font exists -> must train on it
            rest, held = held, []

        for p, font in rest:
            train.append((p, label, font))
        for p, font in held:
            test.append((p, label, font))

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(cfg: DataConfig, train: bool) -> transforms.Compose:
    """
    Train-time augmentation simulates rendering / font variation: small affine
    jitter (rotation, shift, scale) and an occasional perspective warp, with a
    white fill so the background stays consistent with the glyphs.  Test-time is
    deterministic.  Normalisation maps white->+1, black->-1.
    """
    ops: List = [transforms.Grayscale(num_output_channels=cfg.channels)]

    if train and cfg.augment:
        ops += [
            transforms.RandomAffine(
                degrees=6,
                translate=(0.06, 0.06),
                scale=(0.9, 1.1),
                fill=255,                       # white background
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255),
        ]

    # ensure the spatial size is exactly img_size (images are already 128 but be safe)
    ops.append(transforms.Resize((cfg.img_size, cfg.img_size)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.5] * cfg.channels,
                                    std=[0.5] * cfg.channels))

    if train and cfg.augment:
        ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)))

    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TradCharDataset(Dataset):
    """Maps an image path to (tensor, label_index)."""

    def __init__(self, samples: List[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label, _font = self.samples[i]
        img = Image.open(path).convert("L")     # force grayscale
        return self.transform(img), label


class SimplCharEvalDataset(Dataset):
    """
    Maps a *simplified*-character image to the label of its traditional
    counterpart, so a closed-set classifier trained on traditional glyphs can
    be probed for whether it (wrongly) recognises the simplified form too.

    Only covers trad_ids whose `flags` column is exactly `['diff']` (the
    simplified glyph differs from the traditional one, and isn't tangled up
    in a many-to-one merge), and always includes every font.
    """

    def __init__(self, samples: List[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label, _font = self.samples[i]
        img = Image.open(path).convert("L")     # force grayscale
        return self.transform(img), label


def build_simpl_eval_dataset(cfg: DataConfig, id2idx: Dict[str, int]
                             ) -> Tuple[SimplCharEvalDataset, dict]:
    """
    Build the diff-only simplified-character eval set, labelled against an
    existing trained model's `id2idx` so predictions line up with its classes.
    """
    diff_rows = select_diff_only_trad_ids(cfg.csv_path)
    simpl_char_of = {tid: simpl_char for tid, _trad_char, simpl_char in diff_rows}
    images_by_id, dropped_no_image = collect_simpl_images(
        cfg.img_root, diff_rows
    )

    samples: List[Sample] = []
    dropped_not_in_label_space: List[str] = []
    included_tids: List[str] = []
    for tid, imgs in images_by_id.items():
        if tid not in id2idx:
            dropped_not_in_label_space.append(tid)
            continue
        label = id2idx[tid]
        included_tids.append(tid)
        for p, font in imgs:
            samples.append((p, label, font))

    if not samples:
        raise RuntimeError(
            f"No simplified eval images found under {cfg.img_root!r} that "
            "both have a ['diff']-only flag and a trad_id known to id2idx."
        )

    ds = SimplCharEvalDataset(samples, build_transforms(cfg, train=False))
    meta = {
        "n_samples": len(samples),
        "n_trad_ids": len(included_tids),
        "simpl_char_of": simpl_char_of,
        "dropped_no_image": dropped_no_image,
        "dropped_not_in_label_space": dropped_not_in_label_space,
    }
    print(f"[dataset] simpl eval set: {len(samples)} images across "
          f"{len(included_tids)} trad_ids "
          f"(dropped {len(dropped_no_image)} no-image, "
          f"{len(dropped_not_in_label_space)} not in label space)")
    return ds, meta


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    train_ds: TradCharDataset
    test_ds: TradCharDataset
    meta: dict          # num_classes, in_dim, id2idx, idx2id, char_of, counts, cfg...


def build_datasets(cfg: DataConfig) -> DatasetBundle:
    # 1. choose the label space from the CSV (mode-dependent)
    trad_rows = select_trad_ids(cfg.csv_path, cfg.mode)

    # 2. find which of those actually have images on disk
    images_by_id, dropped = collect_images(cfg.img_root, trad_rows)
    if not images_by_id:
        raise RuntimeError(
            f"No traditional images found under {cfg.img_root!r}. "
            "Check data_root / folder layout."
        )

    # 3. build the (deterministic, CSV-ordered) class index over the surviving ids
    included = [tid for tid, _ in trad_rows if tid in images_by_id]
    id2idx = {tid: i for i, tid in enumerate(included)}
    idx2id = {i: tid for tid, i in id2idx.items()}
    char_of = {tid: char for tid, char in trad_rows}

    # 4. split
    if cfg.split_strategy == "stratified_random":
        train_s, test_s = stratified_split(images_by_id, id2idx,
                                           cfg.train_ratio, cfg.seed)
    elif cfg.split_strategy == "leave_font_out":
        train_s, test_s = leave_font_out_split(images_by_id, id2idx,
                                               cfg.holdout_font, cfg.seed)
    else:
        raise ValueError(f"unknown split_strategy {cfg.split_strategy!r}")

    train_ds = TradCharDataset(train_s, build_transforms(cfg, train=True))
    test_ds = TradCharDataset(test_s, build_transforms(cfg, train=False))

    # 5. record exactly which (image, font) ended up in the test set, so a
    #    scoring run can later reproduce per-font evaluation without re-deriving
    #    the split. The manifest is both written to disk and carried in `meta`
    #    (and therefore inside any checkpoint that saves `meta`).
    test_manifest = [
        {"path": p, "trad_id": idx2id[label], "label": label,
         "char": char_of.get(idx2id[label], ""), "font": font}
        for (p, label, font) in test_s
    ]
    os.makedirs(cfg.manifest_dir, exist_ok=True)
    manifest_path = os.path.join(
        cfg.manifest_dir, f"test_manifest_{cfg.mode}_{cfg.split_strategy}.csv"
    )
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "trad_id", "label", "char", "font"]
        )
        writer.writeheader()
        writer.writerows(test_manifest)

    meta = {
        "mode": cfg.mode,
        "num_classes": len(included),
        "channels": cfg.channels,
        "img_size": cfg.img_size,
        "in_dim": cfg.channels * cfg.img_size * cfg.img_size,
        "id2idx": id2idx,
        "idx2id": idx2id,
        "char_of": char_of,
        "split_strategy": cfg.split_strategy,
        "holdout_font": cfg.holdout_font if cfg.split_strategy == "leave_font_out" else None,
        "n_train": len(train_s),
        "n_test": len(test_s),
        "n_dropped_no_image": len(dropped),
        "dropped_ids": dropped,
        "test_manifest": test_manifest,        # per-entry: path, trad_id, label, char, font
        "test_manifest_path": manifest_path,
    }
    print(f"[dataset] wrote test manifest -> {manifest_path}")
    if dropped:
        print(f"[dataset] warning: {len(dropped)} trad_ids had no image and "
              f"were excluded from the label space (first few: {dropped[:5]})")
    print(f"[dataset] mode={cfg.mode}  classes={meta['num_classes']}  "
          f"train={meta['n_train']}  test={meta['n_test']}  "
          f"strategy={cfg.split_strategy}")
    return DatasetBundle(train_ds, test_ds, meta)