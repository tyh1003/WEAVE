import os
import json
import pandas as pd

# 設定路徑
ERROR_OUTPUT_PATH = "data/ALL/Final/errors.jsonl"
RETRIEVE_PATH = "data/ALL/Final/retrieve.jsonl"
REASON_PATH = "data/ALL/Final/reason.jsonl"        # 新增：reason 檔案路徑

# -----------------------------
# 讀取 JSONL 的工具函式
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
# 清洗特定檔案的通用函式
# -----------------------------
def clean_file_by_errors(target_path, error_vids):
    """
    讀取目標路徑檔案，移除 vid 在 error_vids 裡面的資料，並覆蓋回原檔案。
    """
    target_df = read_jsonl(target_path)
    
    if target_df.empty:
        print(f"提示：找不到檔案或檔案為空，跳過處理：{target_path}")
        return

    # 確保 vid 欄位型態為字串
    target_df["vid"] = target_df["vid"].astype(str)
    
    orig_count = len(target_df)
    
    # 篩選：只保留 vid 不在錯誤集合中的資料
    filtered_df = target_df[~target_df["vid"].isin(error_vids)]
    deleted_count = orig_count - len(filtered_df)
    
    # 覆蓋儲存回原本的 JSONL
    filtered_df.to_json(
        target_path, 
        orient="records", 
        lines=True, 
        force_ascii=False
    )
    
    # 取得檔名以方便閱讀 Log
    file_name = os.path.basename(target_path)
    print(f"[{file_name}] 原本筆數: {orig_count} | 刪除筆數: {deleted_count} | 剩餘筆數: {len(filtered_df)}")


# -----------------------------
# 主程式執行流程
# -----------------------------
if __name__ == "__main__":
    print("===== 開始執行資料清洗 =====")
    
    # 1. 讀取錯誤資料並建立 vid 集合
    error_df = read_jsonl(ERROR_OUTPUT_PATH)
    
    if error_df.empty:
        print(f"提示：錯誤檔案不存在或沒有錯誤資料（{ERROR_OUTPUT_PATH}），不需進行刪除。")
    else:
        # 建立錯誤 vid 集合
        error_df["vid"] = error_df["vid"].astype(str)
        error_vids = set(error_df["vid"])
        print(f"偵測到需要刪除的錯誤 vid 總數: {len(error_vids)}\n" + "-"*40)
        
        # 2. 開始清洗 retrieve.jsonl
        clean_file_by_errors(RETRIEVE_PATH, error_vids)
        
        # 3. 開始清洗 reason.jsonl
        clean_file_by_errors(REASON_PATH, error_vids)
        
        print("-"*40 + "\n===== 所有檔案清洗完成 =====")