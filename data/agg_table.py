import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "processed"

trad_df = pd.read_csv(DATA_DIR / "trad_id_freq.csv")
simpl_equiv_df = pd.read_csv(DATA_DIR / "simpl_equivalent.csv")
simpl_freq_df = pd.read_csv(DATA_DIR / "simpl_freq.csv")

merged = trad_df.merge(simpl_equiv_df, on="trad_char", how="left")
merged = merged.merge(simpl_freq_df, on="simpl_char", how="left")

dup_mask = merged.duplicated(subset=["trad_char", "simpl_char"], keep="first")
if dup_mask.any():
    print(f"Found {dup_mask.sum()} duplicate rows on (trad_char, simpl_char) pairs, removing before analysis:")
    print(merged.loc[dup_mask, ["trad_char", "simpl_char"]])
    merged = merged.loc[~dup_mask].reset_index(drop=True)

trad_to_simpl_count = merged.groupby("trad_char")["simpl_char"].nunique()
simpl_to_trad_count = merged.groupby("simpl_char")["trad_char"].nunique()


def compute_flags(row):
    flags = []
    if pd.notna(row["trad_char"]) and pd.notna(row["simpl_char"]) and row["trad_char"] != row["simpl_char"]:
        flags.append("diff")
    if pd.notna(row["simpl_char"]) and simpl_to_trad_count.get(row["simpl_char"], 0) > 1:
        flags.append("many-to-one")
    if pd.notna(row["trad_char"]) and trad_to_simpl_count.get(row["trad_char"], 0) > 1:
        flags.append("one-to-many")
    return flags


merged["flags"] = merged.apply(compute_flags, axis=1)

print("\nFlag summary:")
for flag in ("diff", "many-to-one", "one-to-many"):
    count = merged["flags"].apply(lambda flags, flag=flag: flag in flags).sum()
    print(f"  {flag}: {count} rows")

output_path = DATA_DIR / "trad_simpl_agg.csv"
merged.to_csv(output_path, index=False)

print(f"Wrote {len(merged)} rows to {output_path}")
print("\nMissing value summary per column:")
missing_summary = merged.isna().sum().to_frame(name="missing_count")
missing_summary["missing_pct"] = (missing_summary["missing_count"] / len(merged) * 100).round(2)
print(missing_summary)

merged["trad_freq_rank"] = merged["trad_freq"].rank(ascending=False, method="min")
subset = merged[
    merged["trad_char"].notna()
    & merged["simpl_char"].notna()
    & (merged["trad_freq_rank"] <= 3500)
]

print(f"\nMissing value summary for rows with trad_char & simpl_char present and trad_freq_rank <= 3500 ({len(subset)} rows):")
subset_missing_summary = subset.isna().sum().to_frame(name="missing_count")
subset_missing_summary["missing_pct"] = (subset_missing_summary["missing_count"] / len(subset) * 100).round(2)
subset_missing_summary["total_rows"] = len(subset)
print(subset_missing_summary)

