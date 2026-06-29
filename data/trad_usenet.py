"""
Parse traditional Chinese character usage frequency (Usenet newsgroups during 1993-1994)
and stroke count data into a CSV.
"""

import pandas as pd

INPUT_PATH = "data/raw/trad_usenet_freq.txt"
OUTPUT_PATH = "data/processed/trad_usenet_freq.csv"  # change this to write elsewhere
# TOP_N = 3500

with open(INPUT_PATH, encoding="utf-8") as f:
    rows = []
    for line in f:
        fields = line.split()
        if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
            char, freq, strokes = fields
            rows.append((char, int(freq), int(strokes)))


df = pd.DataFrame(rows, columns=["trad_char", "trad_freq", "trad_stroke_count"])
total_rows = len(df)
print(f'{total_rows} Traditional Chinese characters found')

df = df.sort_values("trad_freq", ascending=False)
# df = df.sort_values("trad_freq", ascending=False).head(TOP_N)
# print(f'Dataset limited to top {TOP_N} frequently used characters ({TOP_N*100/total_rows:.2f}% of original dataset)')

min_freq = df['trad_freq'].min()
max_freq = df['trad_freq'].max()
print(f'Frequency range: {min_freq} to {max_freq}')

print(f'Stroke counts: {df['trad_stroke_count'].min()} to {df['trad_stroke_count'].max()}, Avg={df['trad_stroke_count'].mean():.0f}')

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f'CSV file saved to {OUTPUT_PATH}')