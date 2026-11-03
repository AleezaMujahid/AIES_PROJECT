import pandas as pd

u = pd.read_csv("data/interim/clean_unified.csv.gz", usecols=["source", "project", "commit_id"])
ok = u["commit_id"].astype(str).str.fullmatch(r"[0-9a-f]{40}")

print("commit ids that are NOT a clean 40-character hash, by dataset:")
print(u[~ok].groupby("source").size())
print()
print("examples:")
print(u.loc[~ok, "commit_id"].drop_duplicates().head(10).to_string())
print()
bv = u[u["source"] == "bigvul"]
print("distinct BigVul project names:", bv["project"].nunique())
print(bv["project"].value_counts().head(25).to_string())