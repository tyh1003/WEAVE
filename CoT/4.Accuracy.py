import os
import argparse
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score,  
    recall_score,      
    brier_score_loss,  
    log_loss           
)

# Paths
BASE_DIR = Path("data/ALL")
ALL_PATH = BASE_DIR / "all.jsonl"
PRED_PATH = BASE_DIR / "Final_v3/reason.jsonl"
RETRIEVE_PATH = BASE_DIR / "Final_v3/retrieve.jsonl"    
OUTPUT_PATH = BASE_DIR / "Final_v3/record.jsonl"
ERROR_OUTPUT_PATH = BASE_DIR / "Final_v3/errors.jsonl"
TOKEN_TIME_PATH = BASE_DIR / "Final_v3/token_time.jsonl"

# =====================================================
# 核心修改：只指定這兩個檔案作為成本計算來源
# =====================================================
INPUT_FILES = [
    RETRIEVE_PATH,
    # PRED_PATH
]

# -----------------------------
# Read jsonl
# -----------------------------
def read_jsonl(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# -----------------------------
# Load data
# -----------------------------
all_df = read_jsonl(ALL_PATH)
pred_df = read_jsonl(PRED_PATH)
retrieve_df = read_jsonl(RETRIEVE_PATH)  

# Ensure vid is string (確保三張表的關聯鍵型態一致)
all_df["vid"] = all_df["vid"].astype(str)
pred_df["vid"] = pred_df["vid"].astype(str)
retrieve_df["vid"] = retrieve_df["vid"].astype(str)

# -----------------------------
# Merge dataset info & features
# -----------------------------
merge_df = pred_df.merge(
    all_df[["vid", "dataset"]],
    on="vid",
    how="left"
)

# -----------------------------
# 核心修改：完全排除包含 "FVC" 的所有資料
# -----------------------------
initial_count = len(merge_df)
merge_df = merge_df[~merge_df["dataset"].astype(str).str.contains("FVC", case=False, na=False)]
filtered_count = len(merge_df)
print(f"Filtered out {initial_count - filtered_count} samples containing 'FVC'. Remaining: {filtered_count}")

if merge_df.empty:
    print("Warning: No samples left to evaluate after filtering out 'FVC' datasets.")
    exit()

feature_cols = ["vid"] + [c for c in ["Rc", "Rv", "K_int", "K_ext"] if c in retrieve_df.columns]
merge_df = merge_df.merge(
    retrieve_df[feature_cols],
    on="vid",
    how="left",
    suffixes=('', '_retrieved')
)

for c in ["Rc", "Rv", "K_int", "K_ext"]:
    if f"{c}_retrieved" in merge_df.columns:
        merge_df[c] = merge_df[c].fillna(merge_df[f"{c}_retrieved"])
        merge_df.drop(columns=[f"{c}_retrieved"], inplace=True)

# -----------------------------
# 處理二元標籤與置信度 (conf)
# -----------------------------
merge_df["pred_binary"] = merge_df["pred_label"].astype(int)
merge_df["pred_prob"] = merge_df["conf"].astype(float)
merge_df["label"] = merge_df["label"].astype(int)

# =====================================================
# 核心修改：讀取指定兩個檔案並統計每個 VID 的 Token 與 Time
# =====================================================
print("\nCalculating resource statistics from retrieve_new & reason_new...")
vid_stats = defaultdict(lambda: {"total_token": 0, "total_time": 0.0})

for filepath in INPUT_FILES:
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found, skipping for token/time calc.")
        continue
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            vid = str(data["vid"])
            
            token = data.get("token", 0)
            token = max(0, int(token)) 
            
            time_val = data.get("time", 0.0)
            time_val = max(0.0, float(time_val))
            
            vid_stats[vid]["total_token"] += token
            vid_stats[vid]["total_time"] += time_val

# 儲存每部影片總消耗的 token_time_new.jsonl
token_time_records = []
for vid, stats in vid_stats.items():
    token_time_records.append({
        "vid": vid,
        "total_token": stats["total_token"],
        "total_time": round(stats["total_time"], 4),
    })
os.makedirs(os.path.dirname(TOKEN_TIME_PATH), exist_ok=True)
with open(TOKEN_TIME_PATH, "w", encoding="utf-8") as f:
    for r in token_time_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved: {TOKEN_TIME_PATH}")

# -----------------------------
# Calculate metrics (加入成本計算與 Precision)
# -----------------------------
records = []

def calculate_metrics_dict(name, df_group):
    y_true = df_group["label"]
    y_binary = df_group["pred_binary"]  
    y_prob = df_group["pred_prob"]      
    
    # 計算二元指標 (在此加入 precision)
    acc = accuracy_score(y_true, y_binary)
    f1 = f1_score(y_true, y_binary, zero_division=0)
    prec = precision_score(y_true, y_binary, zero_division=0)
    rec = recall_score(y_true, y_binary, zero_division=0)
    
    # 使用 conf 計算損失指標
    brier = brier_score_loss(y_true, y_prob)
    try:
        logloss = log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        logloss = float('nan')
        
    # 計算該 Group 的 Token 與 Time 平均成本
    group_vids = df_group["vid"].tolist()
    group_tokens = [vid_stats[v]["total_token"] for v in group_vids]
    group_times = [vid_stats[v]["total_time"] for v in group_vids]
    
    avg_token = sum(group_tokens) / len(group_vids) if group_vids else 0
    avg_time = sum(group_times) / len(group_vids) if group_vids else 0
        
    return {
        "dataset": name,
        "num_samples": len(df_group),
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "brier_score": round(float(brier), 4),
        "logloss": round(float(logloss), 4) if not pd.isna(logloss) else "N/A",
        "average_token": round(avg_token, 2),
        "average_time": round(avg_time, 4)
    }

# Per dataset (此處遍歷到的子資料集已自動排除了 FVC)
for dataset_name, group in merge_df.groupby("dataset"):
    if len(group) == 0:
        continue
    records.append(calculate_metrics_dict(dataset_name, group))

# Overall (ALL - 已排除 FVC)
records.append(calculate_metrics_dict("ALL", merge_df))

# -----------------------------
# Save record_new.jsonl
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# 篩選並儲存指定的錯誤資料欄位
# -----------------------------
error_df = merge_df[merge_df["pred_binary"] != merge_df["label"]]
target_cols = ["vid", "dataset", "label", "pred_label", "conf", "Rc", "Rv", "K_int", "K_ext", "reason"]
error_output_df = error_df.reindex(columns=target_cols)

os.makedirs(os.path.dirname(ERROR_OUTPUT_PATH), exist_ok=True)
error_output_df.to_json(
    ERROR_OUTPUT_PATH, 
    orient="records", 
    lines=True,          
    force_ascii=False
)

# -----------------------------
# Print result
# -----------------------------
print("\n===== Evaluation Result =====")
for r in records:
    logloss_str = f"{r['logloss']:.4f}" if isinstance(r['logloss'], float) else str(r['logloss'])
    print(
        f"Dataset: {r['dataset']:<20} "
        f"Samples: {r['num_samples']:<5} | "
        f"Accuracy: {r['accuracy']:.4f} | "
        f"F1: {r['f1']:.4f} | "
        f"Precision: {r['precision']:.4f} | "
        f"Recall: {r['recall']:.4f} | "
        f"Brier: {r['brier_score']:.4f} | "
        f"LogLoss: {logloss_str:<7} | "
        f"AvgToken: {r['average_token']:<9} | "
        f"AvgTime: {r['average_time']:.4f}s"
    )

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Saved mispredicted data to: {ERROR_OUTPUT_PATH} (Total: {len(error_output_df)} rows)")