"""Inspect subfolders that all four fonts have been rendered in traditional Chinese characters."""
import os

IMG_ROOT = "./data/img"          # adjust to your path
FONTS = ["LXGWWenKai", "NotoSansCJK", "NotoSerifCJK", "Xiaolai"]

for tid in sorted(os.listdir(IMG_ROOT)):
    folder = os.path.join(IMG_ROOT, tid)
    if not os.path.isdir(folder):
        continue
    files = set(os.listdir(folder))
    missing = [f for f in FONTS if f"trad_{tid}_{f}.png" not in files]
    if missing:
        print(f"{tid}: missing {missing}")
print('COMPLETE!')