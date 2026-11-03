"""
check_groups.py -- read-only. Shows the biggest commit groups in the BigVul split,
to explain why the vulnerable share is uneven.

Usage:  python Scripts/check_groups.py
"""
import pandas as pd

m = pd.read_csv("data/processed/split_manifest.csv", dtype={"commit_id": str})
m = m[m["source"] == "bigvul"]
m["commit_id"] = m["commit_id"].fillna("(no commit)")
g = (m.groupby("commit_id")
       .agg(rows=("label", "size"), vulnerable=("label", "sum"),
            split=("split", "first"), project=("project_key", "first"))
       .sort_values("rows", ascending=False))
g["pct_vul"] = (100 * g["vulnerable"] / g["rows"]).round(1)

print("Total commit groups:", len(g))
print("\nTop 15 groups by size:")
print(g.head(15).to_string())
print("\nTop 15 groups by vulnerable rows:")
print(g.sort_values("vulnerable", ascending=False).head(15).to_string())
tot = g["rows"].sum()
print("\nShare of all rows held by the top 10 / 50 / 200 groups: "
      f"{g['rows'].head(10).sum()/tot:.0%} / {g['rows'].head(50).sum()/tot:.0%} / {g['rows'].head(200).sum()/tot:.0%}")