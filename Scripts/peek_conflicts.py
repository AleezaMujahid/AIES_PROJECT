import pandas as pd
c = pd.read_csv("data/interim/rejected_conflicts.csv.gz",
                usecols=["dataset", "commit_id", "project", "label", "code_hash", "code"])
for h, g in list(c[c.dataset == "BigVul"].groupby("code_hash"))[:3]:
    print("=" * 60)
    print(g[["commit_id", "project", "label"]].to_string())
    print(g["code"].iloc[0][:400])