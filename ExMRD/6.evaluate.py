import os
import argparse
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,  # 核心修改：引入 Precision 計算工具
    recall_score
)

# =========================================================
# Argument Parsing
# =========================================================
parser = argparse.ArgumentParser(description="Evaluate Misinformation Detection Results and Costs")
parser.add_argument(
    '--data',
    type=str,
    default='ALL',
    help='Dataset folder name (default: ALL)'
)
args = parser.parse_args()

# =========================================================
# Config & Paths
# =========================================================
MODEL_NAME = "gemma-4-31b-it"
BASE_DIR = Path(f"data/{args.data}/CoT/{MODEL_NAME}")

SAVE_PATH = BASE_DIR / "lm_pred.jsonl"
RECORD_PATH = BASE_DIR / "record.jsonl"
TOKEN_TIME_PATH = BASE_DIR / "token_time.jsonl"

INPUT_FILES = [
    "lm_text_refine.jsonl",
    "lm_visual_refine_quad4.jsonl",
    "lm_retrieve_quad4.jsonl",
    "lm_reason.jsonl",
    # "lm_pred.jsonl",
]

# =========================================================
# Utils
# =========================================================
def read_jsonl_keep_vid_as_str(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "vid" in obj:
                obj["vid"] = str(obj["vid"])
            rows.append(obj)
    return pd.DataFrame(rows)

def overwrite_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# =========================================================
# Main Process
# =========================================================
def main():
    # -----------------------------------------------------
    # Step 1: 讀取所有檔案並統計每個 VID 的 Token 與 Time
    # -----------------------------------------------------
    print("Calculating resource statistics (Tokens & Time)...")
    vid_stats = defaultdict(lambda: {"total_token": 0, "total_time": 0.0})
    
    for filename in INPUT_FILES:
        filepath = BASE_DIR / filename
        if not filepath.exists():
            print(f"Warning: File {filename} not found, skipping for token/time calc.")
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

    # 儲存每部影片總消耗的 token_time.jsonl
    token_time_records = []
    for vid, stats in vid_stats.items():
        token_time_records.append({
            "vid": vid,
            "total_token": stats["total_token"],
            "total_time": round(stats["total_time"], 4),
        })
    overwrite_jsonl(TOKEN_TIME_PATH, token_time_records)
    print(f"Saved: {TOKEN_TIME_PATH}")

    # -----------------------------------------------------
    # Step 2: 讀取預測結果並進行模型評估與效能整合
    # -----------------------------------------------------
    print(f"\nLoading predictions from: {SAVE_PATH}")
    pred_df = read_jsonl_keep_vid_as_str(SAVE_PATH)
    
    if pred_df.empty:
        print(f"Error: No prediction results found at {SAVE_PATH}.")
        return

    # 過濾掉 dataset 欄位中包含 "FVC" 的所有資料
    initial_count = len(pred_df)
    pred_df = pred_df[~pred_df["dataset"].astype(str).str.contains("FVC", case=False, na=False)]
    filtered_count = len(pred_df)
    print(f"Filtered out {initial_count - filtered_count} samples containing 'FVC'. Remaining: {filtered_count}")

    if pred_df.empty:
        print("Warning: No samples left to evaluate after filtering out 'FVC' datasets.")
        return

    print("\n========== Evaluation ==========")
    records = []

    # -----------------------------------------------------
    # 子資料集各自評估 (Per Dataset Evaluation)
    # -----------------------------------------------------
    for dataset_name in sorted(pred_df["dataset"].unique()):
        sub_df = pred_df[pred_df["dataset"] == dataset_name]
        
        y_true = sub_df["label"].tolist()
        y_pred = sub_df["pred_label"].tolist()
        
        # 計算模型分類指標 (加入 precision)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        
        # 整合此資料集下的 Token 與 Time 指標
        sub_vids = sub_df["vid"].astype(str).tolist()
        sub_tokens = [vid_stats[v]["total_token"] for v in sub_vids]
        sub_times = [vid_stats[v]["total_time"] for v in sub_vids]
        
        avg_token = sum(sub_tokens) / len(sub_vids) if sub_vids else 0
        avg_time = sum(sub_times) / len(sub_vids) if sub_vids else 0
        
        record = {
            "dataset": dataset_name,
            "num_samples": len(sub_df),
            "accuracy": round(float(accuracy), 6),
            "f1": round(float(f1), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "average_token": round(avg_token, 2),
            "average_time": round(avg_time, 4)
        }
        records.append(record)
        print(record)

    # -----------------------------------------------------
    # 總體評估 (Total Evaluation)
    # -----------------------------------------------------
    y_true_total = pred_df["label"].tolist()
    y_pred_total = pred_df["pred_label"].tolist()
    
    total_accuracy = accuracy_score(y_true_total, y_pred_total)
    total_f1 = f1_score(y_true_total, y_pred_total, average="binary", zero_division=0)
    total_precision = precision_score(y_true_total, y_pred_total, average="binary", zero_division=0)
    total_recall = recall_score(y_true_total, y_pred_total, average="binary", zero_division=0)
    
    # 總體 Token 與 Time 的平均消耗
    all_vids = pred_df["vid"].astype(str).tolist()
    total_tokens = [vid_stats[v]["total_token"] for v in all_vids]
    total_times = [vid_stats[v]["total_time"] for v in all_vids]
    
    global_avg_token = sum(total_tokens) / len(all_vids) if all_vids else 0
    global_avg_time = sum(total_times) / len(all_vids) if all_vids else 0
    
    total_record = {
        "dataset": "TOTAL",
        "num_samples": len(pred_df),
        "accuracy": round(float(total_accuracy), 6),
        "f1": round(float(total_f1), 6),
        "precision": round(float(total_precision), 6),
        "recall": round(float(total_recall), 6),
        "average_token": round(global_avg_token, 2),
        "average_time": round(global_avg_time, 4)
    }
    records.append(total_record)
    print(total_record)

    # -----------------------------------------------------
    # Save Record
    # -----------------------------------------------------
    overwrite_jsonl(RECORD_PATH, records)
    print(f"\nSaved Comprehensive Record -> {RECORD_PATH}")

if __name__ == "__main__":
    main()