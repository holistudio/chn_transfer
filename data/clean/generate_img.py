#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_char_image_dataset.py
==============================

Generate an image dataset of Chinese characters from a CSV, rendering each
character in several fonts.

Output layout
-------------
For every row in the CSV, a subfolder named after the ``trad_id`` column is
created inside OUTPUT_DIR. Inside it:

    <OUTPUT_DIR>/
        A03951/
            trad_A03951_NotoSansCJK.png
            trad_A03951_NotoSerifCJK.png
            trad_A03951_LXGWWenKai.png
            trad_A03951_Xiaolai.png
            simpl_A03951_NotoSansCJK.png      <-- only if the row is flagged "diff"
            simpl_A03951_NotoSerifCJK.png
            simpl_A03951_LXGWWenKai.png
            simpl_A03951_Xiaolai.png
        A02724/
            trad_A02724_NotoSansCJK.png
            ...                               (no simpl_* because trad == simpl)
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
except Exception:  # pragma: no cover - import guard
    _HAVE_FONTTOOLS = False

# tqdm is optional; fall back to a no-op wrapper if it is not installed.
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - import guard
    def tqdm(iterable, **_kwargs):
        return iterable


# =============================================================================
# CONFIG  --  edit these to taste
# =============================================================================

# Paths are anchored to THIS script's location, so they resolve no matter which
# directory you launch Python from (e.g. `python generate_img.py` run from the
# repo root or from data/clean/ both work). ~ is also expanded.
#
# Assumed layout (this script lives in data/clean/):
#     data/
#       clean/
#         trad_simpl_clean.csv   <- CSV_PATH   (next to this script)
#         generate_img.py        <- this script
#       img/                     <- OUTPUT_DIR (sibling of clean/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input CSV: same folder as this script.
CSV_PATH = os.path.join(SCRIPT_DIR, "trad_simpl_clean.csv")
# Output root: the sibling `img/` folder (data/img). Change "../img" if you want
# the dataset somewhere else, e.g. os.path.join(SCRIPT_DIR, "..", "img", "dataset").
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "img"))

# Image geometry / appearance.
IMG_SIZE = 128                 # output images are IMG_SIZE x IMG_SIZE pixels
FONT_SIZE = 96                 # nominal glyph size; ~0.75 * IMG_SIZE leaves margin
SUPERSAMPLE = 2                # render at NxN then downscale (>=2 = crisper glyphs)
BG_COLOR = (255, 255, 255)     # background  (white)
FG_COLOR = (0, 0, 0)           # glyph color (black)
IMAGE_EXT = "png"              # output image format / extension

# Behavior toggles.
SKIP_EXISTING = True           # don't re-render images that already exist (resumable)
CHECK_GLYPH_COVERAGE = True    # skip+log chars a font can't display (needs fonttools)
WRITE_MISSING_LOG = True       # write a CSV of skipped (font, char) pairs into OUTPUT_DIR
VERBOSE = False                # set True for per-character debug logging

# The four fonts. For each:
#   label            -> used verbatim in the file name (MUST contain no spaces)
#   path             -> path to the .ttf / .otf / .ttc on disk (edit to match `fc-list`)
#   index            -> face index inside a .ttc collection (fallback if no match)
#   prefer_subfamily -> for .ttc collections, auto-pick the face whose name
#                       contains this token (e.g. "TC"). Needs fonttools; if it
#                       can't be resolved, `index` is used.
FONTS = [
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
# Implementation  --  generally no need to edit below here
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("char_dataset")


def parse_flags(raw: str) -> list:
    """Parse the stringified-list `flags` cell into a real list.

    Handles '', '[]', "['diff']", "['diff', 'many-to-one']", etc.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
        return list(value) if isinstance(value, (list, tuple)) else [value]
    except (ValueError, SyntaxError):
        # Be forgiving: fall back to a substring check so a malformed cell that
        # still mentions "diff" is not silently treated as "no diff".
        log.warning("Could not parse flags value %r; using substring fallback.", raw)
        return ["diff"] if "diff" in raw else []


def resolve_face_index(path: str, configured_index: int, prefer_subfamily: str | None) -> int:
    """For a .ttc/.otc collection, pick the sub-face whose name contains
    `prefer_subfamily` (e.g. 'TC'). Falls back to `configured_index`."""
    if not prefer_subfamily:
        return configured_index
    if not path.lower().endswith((".ttc", ".otc")):
        return configured_index
    if not _HAVE_FONTTOOLS:
        log.debug("fonttools not available; using index %d for %s", configured_index, path)
        return configured_index

    needle = f" {prefer_subfamily}".lower()
    try:
        collection = TTCollection(path, lazy=True)
        for i, font in enumerate(collection.fonts):
            name_table = font["name"]
            full_name = name_table.getDebugName(4) or name_table.getDebugName(1) or ""
            if needle in f" {full_name}".lower():
                log.debug("Resolved %s face '%s' -> index %d", prefer_subfamily, full_name, i)
                return i
        log.warning(
            "No '%s' sub-face found in %s; using index %d.",
            prefer_subfamily, path, configured_index,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not inspect faces of %s (%s); using index %d.",
                    path, exc, configured_index)
    return configured_index


@lru_cache(maxsize=None)
def load_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    """Cached PIL font loader."""
    return ImageFont.truetype(path, size=size, index=index)


@lru_cache(maxsize=None)
def font_has_glyph(path: str, index: int, char: str) -> bool:
    """True if the font face can display `char`. If fonttools is missing or
    coverage checking is off, this optimistically returns True."""
    if not (_HAVE_FONTTOOLS and CHECK_GLYPH_COVERAGE):
        return True
    try:
        ttfont = TTFont(path, fontNumber=index, lazy=True)
        cmap = ttfont.getBestCmap()
        return ord(char) in cmap
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Glyph check failed for %r in %s (%s); assuming present.", char, path, exc)
        return True


def render_char_image(char: str, font_path: str, face_index: int, out_path: str) -> None:
    """Render a single centered character to `out_path`."""
    ss = max(1, int(SUPERSAMPLE))
    canvas = IMG_SIZE * ss
    font = load_font(font_path, face_index, int(FONT_SIZE * ss))

    img = Image.new("RGB", (canvas, canvas), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # textbbox accounts for the glyph's actual ink box (incl. negative bearings),
    # so we can center precisely rather than relying on the advance width.
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    glyph_w, glyph_h = right - left, bottom - top
    x = (canvas - glyph_w) // 2 - left
    y = (canvas - glyph_h) // 2 - top
    draw.text((x, y), char, font=font, fill=FG_COLOR)

    if ss != 1:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(out_path)


def prepare_fonts() -> list[dict]:
    """Validate configured fonts, expand paths, and resolve TC faces.
    Returns only the fonts that actually exist on disk."""
    usable = []
    for spec in FONTS:
        path = os.path.expanduser(spec["path"])
        label = spec["label"]
        if " " in label:
            log.warning("Font label %r contains spaces; they will appear in file names.", label)
        if not os.path.isfile(path):
            log.warning(
                "Font '%s' not found at %s -- skipping it. "
                "(Install it and/or fix its path in the FONTS config; see the "
                "instructions at the top of this file.)",
                label, path,
            )
            continue
        face_index = resolve_face_index(path, spec.get("index", 0), spec.get("prefer_subfamily"))
        usable.append({"label": label, "path": path, "index": face_index})
        log.info("Using font '%s' (face index %d): %s", label, face_index, path)
    return usable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a per-character image dataset from the Chinese char CSV."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Path to input CSV (overrides CONFIG).")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory (overrides CONFIG).")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process the first N data rows of the CSV (useful for a quick test run).",
    )
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv)
    out_dir = os.path.expanduser(args.out)

    if not os.path.isfile(csv_path):
        log.error("CSV not found: %s", csv_path)
        return 1

    fonts = prepare_fonts()
    if not fonts:
        log.error("No usable fonts found. Install the fonts and fix the FONTS paths, then retry.")
        return 1
    if not (_HAVE_FONTTOOLS and CHECK_GLYPH_COVERAGE):
        log.info("Glyph-coverage checking is OFF (fonttools missing or disabled); "
                 "missing characters may render as blank boxes.")

    os.makedirs(out_dir, exist_ok=True)

    # Counters for the final summary.
    written = skipped_existing = skipped_missing_glyph = skipped_empty = 0
    rows_processed = 0
    missing_records: list[tuple[str, str, str, str]] = []  # (trad_id, kind, char, font_label)
    seen_trad_ids: dict[str, int] = {}  # trad_id -> count of rows claiming it
    rows_with_no_trad_image: list[str] = []  # trad_id of rows where trad_char rendered in zero fonts

    # Count rows first so the progress bar has a total (cheap; file is small).
    with open(csv_path, newline="", encoding="utf-8") as fh:
        total_rows = sum(1 for _ in csv.DictReader(fh))
    if args.limit is not None:
        total_rows = min(total_rows, max(0, args.limit))
        log.info("Limiting run to the first %d data row(s) of the CSV.", args.limit)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"trad_id", "trad_char", "simpl_char", "flags"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            log.error("CSV is missing required column(s): %s", ", ".join(sorted(missing_cols)))
            return 1

        for row_index, row in enumerate(tqdm(reader, total=total_rows, desc="Rendering", unit="row")):
            if args.limit is not None and row_index >= args.limit:
                break
            trad_id = (row.get("trad_id") or "").strip()
            trad_char = (row.get("trad_char") or "").strip()
            simpl_char = (row.get("simpl_char") or "").strip()
            flags = parse_flags(row.get("flags", ""))

            if not trad_id:
                log.warning("Row with empty trad_id skipped: %r", row)
                continue
            if not trad_char:
                log.warning("Row %s has empty trad_char; skipped.", trad_id)
                continue

            rows_processed += 1
            seen_trad_ids[trad_id] = seen_trad_ids.get(trad_id, 0) + 1
            if seen_trad_ids[trad_id] > 1:
                log.warning(
                    "Duplicate trad_id %r seen %d time(s) in the CSV; rows share one "
                    "output folder and earlier images may be overwritten/skipped.",
                    trad_id, seen_trad_ids[trad_id],
                )
            row_dir = os.path.join(out_dir, trad_id)
            os.makedirs(row_dir, exist_ok=True)

            # Build the list of (prefix, character) jobs for this row.
            jobs = [("trad", trad_char)]
            if "diff" in flags:
                if simpl_char:
                    jobs.append(("simpl", simpl_char))
                else:
                    log.warning("Row %s flagged 'diff' but simpl_char is empty; "
                                "no simpl image generated.", trad_id)

            trad_written_for_row = 0
            for prefix, char in jobs:
                for font in fonts:
                    fname = f"{prefix}_{trad_id}_{font['label']}.{IMAGE_EXT}"
                    out_path = os.path.join(row_dir, fname)

                    if SKIP_EXISTING and os.path.exists(out_path):
                        skipped_existing += 1
                        if prefix == "trad":
                            trad_written_for_row += 1
                        continue

                    if not font_has_glyph(font["path"], font["index"], char):
                        skipped_missing_glyph += 1
                        missing_records.append((trad_id, prefix, char, font["label"]))
                        log.debug("'%s' (%s, %s) not in font '%s'; skipped.",
                                  char, prefix, trad_id, font["label"])
                        continue

                    try:
                        render_char_image(char, font["path"], font["index"], out_path)
                        written += 1
                        if prefix == "trad":
                            trad_written_for_row += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        log.warning("Failed to render '%s' (%s/%s) with '%s': %s",
                                    char, trad_id, prefix, font["label"], exc)

            if trad_written_for_row == 0:
                rows_with_no_trad_image.append(trad_id)
                log.warning(
                    "Row %s ('%s') produced zero trad images across all %d font(s); "
                    "its output folder may be empty or missing expected images.",
                    trad_id, trad_char, len(fonts),
                )

    # Optional: dump the list of characters that were skipped for missing glyphs.
    if WRITE_MISSING_LOG and missing_records:
        log_path = os.path.join(out_dir, "_missing_glyphs.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trad_id", "kind", "char", "font_label"])
            w.writerows(missing_records)
        log.info("Wrote missing-glyph log: %s (%d entries)", log_path, len(missing_records))

    duplicate_ids = {tid: n for tid, n in seen_trad_ids.items() if n > 1}
    distinct_folders = len(seen_trad_ids)

    # Summary.
    log.info("-" * 60)
    log.info("Done.")
    log.info("  Rows processed              : %d", rows_processed)
    log.info("  Distinct trad_id folders    : %d", distinct_folders)
    log.info("  Fonts used                  : %s", ", ".join(f["label"] for f in fonts))
    log.info("  Images written              : %d", written)
    log.info("  Skipped (already existed)   : %d", skipped_existing)
    log.info("  Skipped (missing glyph)     : %d", skipped_missing_glyph)
    log.info("  Output directory            : %s", os.path.abspath(out_dir))
    if duplicate_ids:
        log.warning(
            "Found %d duplicate trad_id value(s) accounting for the gap between "
            "rows processed (%d) and distinct folders (%d): %s",
            len(duplicate_ids), rows_processed, distinct_folders,
            ", ".join(f"{tid} x{n}" for tid, n in duplicate_ids.items()),
        )
    if rows_with_no_trad_image:
        log.warning(
            "%d row(s) ended up with zero trad images written/kept (folder may be "
            "empty): %s",
            len(rows_with_no_trad_image), ", ".join(rows_with_no_trad_image),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())