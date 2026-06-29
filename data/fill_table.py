from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "processed"
INPUT_PATH = DATA_DIR / "trad_simpl_agg.csv"
OUTPUT_PATH = DATA_DIR / "trad_simpl_filled.csv"
OPENCC_URL = "https://raw.githubusercontent.com/BYVoid/OpenCC/master/data/dictionary/TSCharacters.txt"

response = requests.get(OPENCC_URL)
response.raise_for_status()

trad_to_simpl = {}
for line in response.text.splitlines():
    if not line or line.startswith("#"):
        continue
    trad_char, simpl_chars = line.split("\t")
    trad_to_simpl[trad_char] = simpl_chars.split()

simpl_to_trad = {}
for trad_char, simpl_chars in trad_to_simpl.items():
    for simpl_char in simpl_chars:
        simpl_to_trad.setdefault(simpl_char, []).append(trad_char)

df = pd.read_csv(INPUT_PATH)
# trad_char/simpl_char load as a non-nullable "str" dtype here, which rejects list values -
# cast to object so the OpenCC lookups below (which assign lists, then explode them) can be stored.
df["trad_char"] = df["trad_char"].astype(object)
df["simpl_char"] = df["simpl_char"].astype(object)

missing_simpl_mask = df["trad_char"].notna() & df["trad_id"].notna() & df["simpl_char"].isna()
df.loc[missing_simpl_mask, "simpl_char"] = df.loc[missing_simpl_mask, "trad_char"].map(trad_to_simpl)
filled_simpl_count = (df.loc[missing_simpl_mask, "simpl_char"].notna()).sum()
df = df.explode("simpl_char", ignore_index=True)

missing_trad_mask = df["simpl_char"].notna() & df["trad_char"].isna()
df.loc[missing_trad_mask, "trad_char"] = df.loc[missing_trad_mask, "simpl_char"].map(simpl_to_trad)
filled_trad_count = (df.loc[missing_trad_mask, "trad_char"].notna()).sum()
df = df.explode("trad_char", ignore_index=True)

print(f"Filled simpl_char via OpenCC for {filled_simpl_count} rows (trad_char present, simpl_char missing).")
print(f"Filled trad_char via OpenCC for {filled_trad_count} rows (simpl_char present, trad_char missing).")

trad_to_simpl_count = df.groupby("trad_char")["simpl_char"].nunique()
simpl_to_trad_count = df.groupby("simpl_char")["trad_char"].nunique()


def compute_flags(row):
    flags = []
    if pd.notna(row["trad_char"]) and pd.notna(row["simpl_char"]) and row["trad_char"] != row["simpl_char"]:
        flags.append("diff")
    if pd.notna(row["simpl_char"]) and simpl_to_trad_count.get(row["simpl_char"], 0) > 1:
        flags.append("many-to-one")
    if pd.notna(row["trad_char"]) and trad_to_simpl_count.get(row["trad_char"], 0) > 1:
        flags.append("one-to-many")
    return flags


df["flags"] = df.apply(compute_flags, axis=1)

print("\nFlag summary:")
for flag in ("diff", "many-to-one", "one-to-many"):
    count = df["flags"].apply(lambda flags, flag=flag: flag in flags).sum()
    print(f"  {flag}: {count} rows")

df.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")

print("\nMissing value summary per column:")
missing_summary = df.isna().sum().to_frame(name="missing_count")
missing_summary["missing_pct"] = (missing_summary["missing_count"] / len(df) * 100).round(2)
print(missing_summary)

df["trad_freq_rank"] = df["trad_freq"].rank(ascending=False, method="min")
subset = df[
    df["trad_char"].notna()
    & df["simpl_char"].notna()
    & (df["trad_freq_rank"] <= 3500)
]

print(f"\nMissing value summary for rows with trad_char & simpl_char present and trad_freq_rank <= 3500 ({len(subset)} rows):")
subset_missing_summary = subset.isna().sum().to_frame(name="missing_count")
subset_missing_summary["missing_pct"] = (subset_missing_summary["missing_count"] / len(subset) * 100).round(2)
subset_missing_summary["total_rows"] = len(subset)
print(subset_missing_summary)
