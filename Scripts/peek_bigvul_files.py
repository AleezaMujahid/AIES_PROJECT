import glob, os
import pandas as pd

files = []
for f in glob.glob("data/raw/bigvul/**/*.csv", recursive=True):
    if "__MACOSX" in f or os.path.basename(f).startswith("._"):
        continue
    files.append(f)

for f in sorted(files):
    print("\n==", f)
    if os.path.getsize(f) > 300 * 1024 * 1024:
        print("big file, header only:", pd.read_csv(f, nrows=0).columns.tolist())
        continue
    df = pd.read_csv(f, low_memory=False)
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())