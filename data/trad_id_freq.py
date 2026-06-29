"""
Merge character IDs and Unicode codepoints into the frequency CSV.
"""

import pandas as pd

FREQ_CSV_PATH = "data/processed/trad_usenet_freq.csv"
MINISTRY_TSV_PATH = "data/raw/trad_ministry_id.tsv"
OUTPUT_PATH = "data/processed/trad_id_freq.csv"  # change this to write elsewhere

freq_df = pd.read_csv(FREQ_CSV_PATH)

ministry_df = pd.read_csv(MINISTRY_TSV_PATH, sep="\t", skiprows=1, dtype=str)
ministry_df = ministry_df.rename(columns={"教育部字號": "trad_id", "Unicode": "trad_unicode", "常用字": "trad_char"})
ministry_df = ministry_df[["trad_char", "trad_id", "trad_unicode"]]

merged_df = freq_df.merge(ministry_df, on="trad_char", how="left")

merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"CSV file saved to {OUTPUT_PATH}")

unmatched = merged_df.loc[merged_df["trad_id"].isna(), "trad_char"]
print(f"{len(unmatched)} characters had no matching ministry id")
print("10 random unmatched characters:", unmatched.sample(min(10, len(unmatched))).tolist())