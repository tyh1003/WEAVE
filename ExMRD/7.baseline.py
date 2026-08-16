import pandas as pd

# =========================================================
# 1. 建立資料集樣本數與模型指標的資料表
# =========================================================
# 樣本數資料 (來自第一張表)
sample_counts = {
    "FakeSV": 3624,
    "FakeTT": 1991,
    "FVC": 2764
}

# 模型預測指標資料 (來自第二張表，數值轉換為 0~1 浮點數)
model_performance = {
    "FakeSV": {"ACC": 0.8690, "F1": 0.8652, "P": 0.8731, "R": 0.8613},
    "FakeTT": {"ACC": 0.8428, "F1": 0.8313, "P": 0.8227, "R": 0.8519},
    "FVC":    {"ACC": 0.9682, "F1": 0.9675, "P": 0.9702, "R": 0.9675}
}

# =========================================================
# 2. 排除 FVC 並進行加權總和計算 (Weighted Average)
# =========================================================
# 指定要納入計算的資料集清單
target_datasets = ["FakeSV", "FakeTT"]

total_samples = 0
weighted_acc = 0.0
weighted_f1 = 0.0
weighted_p = 0.0
weighted_r = 0.0

for dataset in target_datasets:
    n = sample_counts[dataset]
    metrics = model_performance[dataset]
    
    total_samples += n
    weighted_acc += metrics["ACC"] * n
    weighted_f1 += metrics["F1"] * n
    weighted_p += metrics["P"] * n
    weighted_r += metrics["R"] * n

# 計算加權平均值
total_accuracy = weighted_acc / total_samples
total_f1 = weighted_f1 / total_samples
total_precision = weighted_p / total_samples
total_recall = weighted_r / total_samples

# =========================================================
# 3. 輸出結果
# =========================================================
print("========== Evaluation Results (Excluding FVC) ==========")
print(f"Total Samples evaluated: {total_samples}")
print(f"Total Accuracy  : {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
print(f"Total F1-Score  : {total_f1:.4f} ({total_f1*100:.2f}%)")
print(f"Total Precision : {total_precision:.4f} ({total_precision*100:.2f}%)")
print(f"Total Recall    : {total_recall:.4f} ({total_recall*100:.2f}%)")