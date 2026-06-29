#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_char_image_dataset.py
==============================

Generate an image dataset of Chinese characters from a CSV, rendering each
character in several fonts. Supports both flat and nested font configurations.
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import os
import sys
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

# fonttools is optional; the script degrades gracefully without it.
try:
    from fontTools.ttLib import TTFont, TTCollection  # noqa: F401
    _HAVE_FONTTOOLS = True
except Exception:
    _HAVE_FONTTOOLS = False

# tqdm is optional; fall back to a no-op wrapper if it is not installed.
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_kwargs):
        return iterable


# =============================================================================
# CONFIG  
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "trad_simpl_clean.csv")
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "img"))

IMG_SIZE = 128
FONT_SIZE = 96
SUPERSAMPLE = 2
BG_COLOR = (255, 255, 255)
FG_COLOR = (0, 0, 0)
IMAGE_EXT = "png"

SKIP_EXISTING = True
CHECK_GLYPH_COVERAGE = True
WRITE_MISSING_LOG = True
VERBOSE = False

# Paths updated to match your WSL2 environment based on fc-list
FONTS = [
    {
        "label": "ARPLUKai",
        "trad":  {"path": "/usr/share/fonts/truetype/arphic/ukai.ttc", "prefer_subfamily": "TW"},
        "simpl": {"path": "/usr/share/fonts/truetype/arphic/ukai.ttc", "prefer_subfamily": "CN"},
    },
    {
        "label": "ARPLUMing",
        "trad":  {"path": "/usr/share/fonts/truetype/arphic/uming.ttc", "prefer_subfamily": "TW"},
        "simpl": {"path": "/usr/share/fonts/truetype/arphic/uming.ttc", "prefer_subfamily": "CN"},
    },
    # {
    #     "label": "LXGWBright",
    #     "trad":  {"path": "~/.local/share/fonts/LXGWBrightTC-Regular.ttf"},
    #     "simpl": {"path": "~/.local/share/fonts/LXGWBrightGB-Regular.ttf"},
    # },
    {
        "label": "HanyiSentyWen",
        "trad":  {"path": "~/.local/share/fonts/SentyWEN2017.ttf"},
        "simpl": {"path": "~/.local/share/fonts/SentyWEN2017.ttf"},
    },
    {
        "label": "HanyiSentyTangType",
        "trad":  {"path": "~/.local/share/fonts/HanyiSentyTang.ttf"},
        "simpl": {"path": "~/.local/share/fonts/HanyiSentyTang.ttf"},
    },
    {
        "label": "MaokenAssortedSans",
        "trad":  {"path": "~/.local/share/fonts/MaokenAssortedSans-TC.ttf"},
        "simpl": {"path": "~/.local/share/fonts/MaokenAssortedSans.ttf"},
    },
    {
        "label": "GNUUnifont",
        "trad":  {"path": "/usr/share/fonts/opentype/unifont/unifont.otf"},
        "simpl": {"path": "/usr/share/fonts/opentype/unifont/unifont.otf"},
    },
    # Original flat schemas
    {
        "label": "NotoSansCJK",
        "path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "index": 0,
        "prefer_subfamily": "TC",
    },
    {
        "label": "NotoSerifCJK",
        "path": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "index": 0,
        "prefer_subfamily": "TC",
    },
    {
        "label": "LXGWWenKai",
        "path": "~/.local/share/fonts/LXGWWenKai-Regular.ttf",
        "index": 0,
        "prefer_subfamily": None,
    },
    {
        "label": "Xiaolai",
        "path": "~/.local/share/fonts/XiaolaiSC-Regular.ttf",
        "index": 0,
        "prefer_subfamily": None,
    },
]

# =============================================================================
# Implementation  
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("char_dataset")

def parse_flags(raw: str) -> list:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
        return list(value) if isinstance(value, (list, tuple)) else [value]
    except (ValueError, SyntaxError):
        log.warning("Could not parse flags value %r; using substring fallback.", raw)
        return ["diff"] if "diff" in raw else []

def resolve_face_index(path: str, configured_index: int, prefer_subfamily: str | None) -> int:
    if not prefer_subfamily:
        return configured_index
    if not path.lower().endswith((".ttc", ".otc")):
        return configured_index
    if not _HAVE_FONTTOOLS:
        return configured_index

    needle = f" {prefer_subfamily}".lower()
    try:
        collection = TTCollection(path, lazy=True)
        for i, font in enumerate(collection.fonts):
            name_table = font["name"]
            full_name = name_table.getDebugName(4) or name_table.getDebugName(1) or ""
            if needle in f" {full_name}".lower():
                return i
    except Exception as exc:
        log.warning("Could not inspect faces of %s (%s); using index %d.", path, exc, configured_index)
    return configured_index

@lru_cache(maxsize=None)
def load_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)

@lru_cache(maxsize=None)
def font_has_glyph(path: str, index: int, char: str) -> bool:
    if not (_HAVE_FONTTOOLS and CHECK_GLYPH_COVERAGE):
        return True
    try:
        ttfont = TTFont(path, fontNumber=index, lazy=True)
        cmap = ttfont.getBestCmap()
        return ord(char) in cmap
    except Exception:
        return True

def render_char_image(char: str, font_path: str, face_index: int, out_path: str) -> None:
    ss = max(1, int(SUPERSAMPLE))
    canvas = IMG_SIZE * ss
    font = load_font(font_path, face_index, int(FONT_SIZE * ss))

    img = Image.new("RGB", (canvas, canvas), BG_COLOR)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    glyph_w, glyph_h = right - left, bottom - top
    x = (canvas - glyph_w) // 2 - left
    y = (canvas - glyph_h) // 2 - top
    draw.text((x, y), char, font=font, fill=FG_COLOR)

    if ss != 1:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(out_path)

def prepare_fonts() -> list[dict]:
    """Validate configured fonts and parse both flat and nested dictionary schemas."""
    usable = []
    for spec in FONTS:
        label = spec["label"]
        if " " in label:
            log.warning("Font label %r contains spaces; they will appear in file names.", label)

        font_config = {"label": label, "trad": None, "simpl": None}

        def resolve_side(side_key: str):
            # Handle flat schema (original)
            if "path" in spec and side_key not in spec:
                path = os.path.expanduser(spec["path"])
                if os.path.isfile(path):
                    idx = spec.get("index", 0)
                    subfam = spec.get("prefer_subfamily")
                    return {"path": path, "index": resolve_face_index(path, idx, subfam)}
                return None

            # Handle nested schema (new additions)
            if side_key in spec:
                side_spec = spec[side_key]
                path = side_spec.get("path") or spec.get("path")
                
                if not path:
                    log.warning("Font '%s' (%s) is missing a 'path' key.", label, side_key)
                    return None

                path = os.path.expanduser(path)
                if os.path.isfile(path):
                    idx = side_spec.get("index", spec.get("index", 0))
                    subfam = side_spec.get("prefer_subfamily", spec.get("prefer_subfamily"))
                    return {"path": path, "index": resolve_face_index(path, idx, subfam)}
                else:
                    log.warning("Font file not found for '%s' (%s): %s", label, side_key, path)
                    return None
            return None

        font_config["trad"] = resolve_side("trad")
        font_config["simpl"] = resolve_side("simpl")

        if font_config["trad"] or font_config["simpl"]:
            usable.append(font_config)
            log.info("Loaded font '%s'", label)
        else:
            log.warning("Font '%s' skipped (missing valid paths for both trad and simpl).", label)

    return usable

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv)
    out_dir = os.path.expanduser(args.out)

    if not os.path.isfile(csv_path):
        log.error("CSV not found: %s", csv_path)
        return 1

    fonts = prepare_fonts()
    if not fonts:
        log.error("No usable fonts found.")
        return 1

    os.makedirs(out_dir, exist_ok=True)

    written = skipped_existing = skipped_missing_glyph = 0
    rows_processed = 0
    missing_records = []
    seen_trad_ids = {}
    rows_with_no_trad_image = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        total_rows = sum(1 for _ in csv.DictReader(fh))
    if args.limit is not None:
        total_rows = min(total_rows, max(0, args.limit))

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_index, row in enumerate(tqdm(reader, total=total_rows, desc="Rendering", unit="row")):
            if args.limit is not None and row_index >= args.limit:
                break
            
            trad_id = (row.get("trad_id") or "").strip()
            trad_char = (row.get("trad_char") or "").strip()
            simpl_char = (row.get("simpl_char") or "").strip()
            flags = parse_flags(row.get("flags", ""))

            if not trad_id or not trad_char:
                continue

            rows_processed += 1
            seen_trad_ids[trad_id] = seen_trad_ids.get(trad_id, 0) + 1
            row_dir = os.path.join(out_dir, trad_id)
            os.makedirs(row_dir, exist_ok=True)

            jobs = [("trad", trad_char)]
            if "diff" in flags and simpl_char:
                jobs.append(("simpl", simpl_char))

            trad_written_for_row = 0
            for prefix, char in jobs:
                for font in fonts:
                    side_config = font[prefix]
                    if not side_config:
                        continue 

                    fpath = side_config["path"]
                    findex = side_config["index"]
                    fname = f"{prefix}_{trad_id}_{font['label']}.{IMAGE_EXT}"
                    out_path = os.path.join(row_dir, fname)

                    if SKIP_EXISTING and os.path.exists(out_path):
                        skipped_existing += 1
                        if prefix == "trad":
                            trad_written_for_row += 1
                        continue

                    if not font_has_glyph(fpath, findex, char):
                        skipped_missing_glyph += 1
                        missing_records.append((trad_id, prefix, char, font["label"]))
                        continue

                    try:
                        render_char_image(char, fpath, findex, out_path)
                        written += 1
                        if prefix == "trad":
                            trad_written_for_row += 1
                    except Exception as exc:
                        log.warning("Failed to render '%s' (%s/%s) with '%s': %s",
                                    char, trad_id, prefix, font["label"], exc)

            if trad_written_for_row == 0:
                rows_with_no_trad_image.append(trad_id)

    if WRITE_MISSING_LOG and missing_records:
        log_path = os.path.join(out_dir, "_missing_glyphs.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trad_id", "kind", "char", "font_label"])
            w.writerows(missing_records)

    log.info("-" * 60)
    log.info("Done.")
    log.info("  Rows processed              : %d", rows_processed)
    log.info("  Images written              : %d", written)
    log.info("  Skipped (already existed)   : %d", skipped_existing)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())