"""Parse simplified Chinese character frequency data into a CSV."""
import pandas as pd

INPUT_PATH = "data/raw/sim_modern_freq.tsv"
OUTPUT_PATH = "data/processed/simpl_freq.csv"  # change this to write elsewhere

df = pd.read_csv(INPUT_PATH, sep="\t", skiprows=1, header=None, names=["rank", "simpl_char", "simpl_freq"])
df = df[["simpl_char", "simpl_freq"]]

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"{len(df)} rows saved to {OUTPUT_PATH}")
