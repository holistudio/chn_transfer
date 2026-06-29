import ast
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(exist_ok=True)

TOP_N = 3500

df = pd.read_csv(DATA_DIR / "processed" / "trad_simpl_agg.csv")

# df = df.dropna(subset=["trad_id", "trad_unicode"])
df = df.dropna()
df = df.sort_values("trad_freq", ascending=False)
df = df.head(TOP_N)

def quote_field(value, force_quote=False):
    if pd.isna(value):
        return ""
    s = str(value)
    needs_quote = force_quote or any(c in s for c in (",", '"', "\n", "\r"))
    if needs_quote:
        return '"' + s.replace('"', '""') + '"'
    return s


output_path = CLEAN_DIR / "trad_simpl_clean.csv"
with open(output_path, "w", encoding="utf-8", newline="") as f:
    f.write(",".join(df.columns) + "\n")
    for _, row in df.iterrows():
        fields = [
            quote_field(row[col], force_quote=(col == "english"))
            for col in df.columns
        ]
        f.write(",".join(fields) + "\n")

print(f"Wrote {len(df)} rows to {output_path}")

parsed_flags = df["flags"].apply(ast.literal_eval)
print("\nFlag summary (% of rows in trad_simpl_clean.csv):")
for flag in ("diff", "many-to-one", "one-to-many"):
    count = parsed_flags.apply(lambda flags, flag=flag: flag in flags).sum()
    pct = round(count / len(df) * 100, 2) if len(df) else 0.0
    print(f"  {flag}: {count} rows ({pct}%)")

diff_mask = parsed_flags.apply(lambda flags: "diff" in flags)
diff_flags = parsed_flags[diff_mask]
diff_count = len(diff_flags)
print(f"\nFlag summary among rows with 'diff' flag ({diff_count} rows), normalized to diff row count:")
for flag in ("many-to-one", "one-to-many"):
    count = diff_flags.apply(lambda flags, flag=flag: flag in flags).sum()
    pct = round(count / diff_count * 100, 2) if diff_count else 0.0
    print(f"  {flag}: {count} rows ({pct}%)")
