import pandas as pd
import json

# =========================================================
# Path
# =========================================================

PATH_1 = "data/ALL/refine_retireve/refine_retireve.jsonl"
PATH_2 = "data/ALL/refine/Rc.jsonl"

OUTPUT_PATH = "data/ALL/refine/Rc_diff.jsonl"

# =========================================================
# Load jsonl
# =========================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)

df1 = load_jsonl(PATH_1)
df2 = load_jsonl(PATH_2)

# =========================================================
# Rename columns
# =========================================================

df1 = df1.rename(columns={
    "Rc": "Rc_refine_retrieve"
})

df2 = df2.rename(columns={
    "Rc": "Rc_refine"
})

# =========================================================
# Merge by vid
# =========================================================

merged = pd.merge(
    df1[["vid", "Rc_refine_retrieve"]],
    df2[["vid", "Rc_refine"]],
    on="vid",
    how="inner"
)

# =========================================================
# Compare Rc
# =========================================================

merged["is_same"] = (
    merged["Rc_refine_retrieve"].fillna("").str.strip()
    ==
    merged["Rc_refine"].fillna("").str.strip()
)

# 只保留不同的資料
diff_df = merged[~merged["is_same"]].copy()

# =========================================================
# Save jsonl
# =========================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in diff_df.to_dict(orient="records"):
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# =========================================================
# Result
# =========================================================

print(f"Total merged: {len(merged)}")
print(f"Different Rc: {len(diff_df)}")
print(f"Saved to: {OUTPUT_PATH}")