"""Scrape the Wikisource list of frequently used Simplified/Traditional Chinese characters into a CSV."""
import re
from html.parser import HTMLParser

import pandas as pd
import requests

URL = "https://en.wikisource.org/wiki/Translation:List_of_Frequently_Used_Characters_in_Modern_Chinese"
OUTPUT_PATH = "data/processed/simpl_equivalent.csv"  # change this to write elsewhere
START_SECTION_ID = "List_of_characters"


class CharacterTableParser(HTMLParser):
    """Walks the article body, only collecting <table> rows inside the "List of characters" h2 section."""

    def __init__(self):
        super().__init__()
        self.rows = []  # list of dicts: simpl_char, simpl_stroke_count, trad_char, english

        self.in_target_section = False
        self.stopped = False
        self.stroke_count = None

        self.heading_tag = None  # set while inside an h2/h3 tag, to capture its text
        self.heading_text = ""

        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_text = ""
        self.row_cells = []  # accumulated cell texts for the current <tr>

        self.current_simpl_char = None

    def handle_starttag(self, tag, attrs):
        if self.stopped:
            return

        attrs_dict = dict(attrs)

        if tag == "h2":
            if self.in_target_section:
                # Reached the next h2 (e.g. "Inferior frequently used characters") - stop entirely.
                self.stopped = True
                return
            if attrs_dict.get("id") == START_SECTION_ID:
                self.in_target_section = True
            self.heading_tag = tag
            self.heading_text = ""
            return

        if not self.in_target_section:
            return

        if tag == "h3":
            self.heading_tag = tag
            self.heading_text = ""
        elif tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.row_cells = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.cell_text = ""

    def handle_endtag(self, tag):
        if self.stopped:
            return

        if tag in ("h2", "h3") and self.heading_tag == tag:
            if tag == "h3":
                match = re.match(r"\s*(\d+)", self.heading_text)
                if match:
                    self.stroke_count = int(match.group(1))
            self.heading_tag = None
        elif tag == "table":
            self.in_table = False
        elif tag == "tr" and self.in_table:
            self.in_row = False
            self._process_row()
        elif tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.row_cells.append(self.cell_text.strip())

    def handle_data(self, data):
        if self.heading_tag is not None:
            self.heading_text += data
        if self.in_cell:
            self.cell_text += data

    def _process_row(self):
        cells = self.row_cells
        if len(cells) == 4:
            # New character group: Simp. | Trad. | Pinyin | English
            simpl_char, trad_text, _pinyin, english = cells
            if simpl_char == "Simp.":
                return  # header row
            self.current_simpl_char = simpl_char
        elif len(cells) == 3:
            # Continuation row sharing the simp char from the previous rowspan group: Trad. | Pinyin | English
            trad_text, _pinyin, english = cells
        else:
            return

        for trad_char in trad_text.split(","):
            trad_char = trad_char.strip()
            if trad_char:
                self.rows.append({
                    "simpl_char": self.current_simpl_char,
                    "simpl_stroke_count": self.stroke_count,
                    "trad_char": trad_char,
                    "english": english.strip(),
                })


response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
response.raise_for_status()

parser = CharacterTableParser()
parser.feed(response.text)

df = pd.DataFrame(parser.rows, columns=["simpl_char", "simpl_stroke_count", "trad_char", "english"])
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"{len(df)} rows saved to {OUTPUT_PATH}")
print(f'Stroke counts: {df['simpl_stroke_count'].min()} to {df['simpl_stroke_count'].max()}, Avg={df['simpl_stroke_count'].mean():.0f}')
