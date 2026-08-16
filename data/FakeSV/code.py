import os
import json
import pandas as pd

# ========== 設定路徑 ==========
src_jsonl = './data/FakeSV/data_complete.json'
dst_jsonl = './data/FakeSV/data.jsonl'
dst_label_jsonl = './data/FakeSV/label.jsonl'

# ========== 1. 載入完整資料 ==========
with open(src_jsonl, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

# ========== 2. key 對應與重命名 ==========
key_map = {
    "video_id": "vid",
    "annotation": "label"
}
def rename_keys(d, key_map):
    return {key_map.get(k, k): v for k, v in d.items()}

new_data = [rename_keys(d, key_map) for d in data]

# ========== 3. 輸出新的 data.jsonl ==========
with open(dst_jsonl, 'w', encoding='utf-8') as f:
    for d in new_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print('data.jsonl 完成，數量:', len(new_data))

# ========== 4. 產生 label.jsonl ==========
df = pd.read_json(dst_jsonl, lines=True, dtype={'vid': str})

label_map = {
    '假': 0,
    '真': 1,
    '辟谣': 2  # 注意簡體
}
df_label = df[['vid', 'label']].copy()
df_label['label'] = df_label['label'].map(label_map)
df_label = df_label.dropna(subset=['label'])

df_label.to_json(dst_label_jsonl, lines=True, orient='records', force_ascii=False)
print('label.jsonl 完成，數量:', len(df_label))
