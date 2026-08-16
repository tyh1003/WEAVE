import os
import json
import pandas as pd

# === 1. 讀取 ./data/FakeTT/data.json（每行一筆的 jsonl 檔）===
src_json = './data/FakeTT/data.json'
dst_jsonl = './data/FakeTT/data.jsonl'
dst_label_jsonl = './data/FakeTT/label.jsonl'

# 讀入原始 data.json
with open(src_json, 'r', encoding='utf-8') as f:
    # 判斷格式是 JSON lines or list
    first = f.read(2)
    f.seek(0)
    if first == '[\n' or first == '[':
        # list 格式
        data = json.load(f)
    else:
        # 一行一筆
        data = [json.loads(line) for line in f if line.strip()]

# key 對應
key_map = {
    "video_id": "vid",
    "annotation": "label"
}
def rename_keys(d, key_map):
    return {key_map.get(k, k): v for k, v in d.items()}

new_data = [rename_keys(d, key_map) for d in data]

# 存成 data.jsonl
with open(dst_jsonl, 'w', encoding='utf-8') as f:
    for d in new_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print('data.jsonl 完成，數量:', len(new_data))

# label 對應
label_map = {
    'fake': 1,
    'real': 0
}
df = pd.read_json(dst_jsonl, lines=True, dtype={'vid': str})
df_label = df[['vid', 'label']].copy()
df_label['label'] = df_label['label'].map(label_map)
df_label = df_label.dropna(subset=['label'])

df_label.to_json(dst_label_jsonl, lines=True, orient='records', force_ascii=False)
print('label.jsonl 完成，數量:', len(df_label))
