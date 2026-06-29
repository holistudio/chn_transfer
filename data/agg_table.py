import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "processed"

trad_df = pd.read_csv(DATA_DIR / "trad_id_freq.csv")
simpl_equiv_df = pd.read_csv(DATA_DIR / "simpl_equivalent.csv")
simpl_freq_df = pd.read_csv(DATA_DIR / "simpl_freq.csv")

merged = trad_df.merge(simpl_equiv_df, on="trad_char", how="left")
merged = merged.merge(simpl_freq_df, on="simpl_char", how="left")

# simpl_equiv_df rows with no trad_char (simplified chars shown with no traditional equivalent)
# never match anything in trad_df's trad_char-keyed merge above, so add them in separately.
no_trad_df = simpl_equiv_df[simpl_equiv_df["trad_char"].isna()].merge(simpl_freq_df, on="simpl_char", how="left")
merged = pd.concat([merged, no_trad_df], ignore_index=True)

dup_mask = merged.duplicated(subset=["trad_char", "simpl_char"], keep="first")
if dup_mask.any():
    print(f"Found {dup_mask.sum()} duplicate rows on (trad_char, simpl_char) pairs, removing before analysis:")
    print(merged.loc[dup_mask, ["trad_char", "simpl_char"]])
    merged = merged.loc[~dup_mask].reset_index(drop=True)

output_path = DATA_DIR / "trad_simpl_agg.csv"
merged.to_csv(output_path, index=False)

print(f"Wrote {len(merged)} rows to {output_path}")

