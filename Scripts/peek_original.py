import pandas as pd

df = pd.read_csv("data/raw/bigvul/all_c_cpp_release2.0.csv", low_memory=False)
print("shape:", df.shape)
print(df["lang"].value_counts().head(10))
print("duplicate cve+commit rows:", df.duplicated(subset=["cve_id", "commit_id"]).sum())
print("missing files_changed:", df["files_changed"].isna().sum())

s = df["files_changed"].dropna().astype(str)
print("files_changed length in characters, median / max:", s.str.len().median(), s.str.len().max())
print("--- first 1500 characters of one sample ---")
print(s.iloc[0][:1500])