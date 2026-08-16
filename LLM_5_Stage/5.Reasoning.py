import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import re

# 🔥 新 SDK
from google import genai

# =========================
# 參數設定
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='FakeSV')
parser.add_argument('--model', type=str, default='gemini-3.1-flash-lite-preview')
parser.add_argument('--top_n', type=int, default=335)
args = parser.parse_args()

dataset = args.data
model_name = args.model
TOP_N = args.top_n

# input_path = f"data/{dataset}/CoT/{model_name}/retrieve.jsonl"
# clip_path = f"data/{dataset}/CoT/{model_name}/CLIP.jsonl"

input_path = "retrieve.jsonl"
clip_path = "CLIP.jsonl"


# =========================
# 讀 JSONL
# =========================
def read_jsonl_safe(path):
    
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                obj["vid"] = str(obj["vid"])
                rows.append(obj)
            except:
                continue
    return rows


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "reason_0425.jsonl")

# =========================
# 🔥 清空輸出
# =========================
open(save_path, "w").close()

# =========================
# 初始化 LLM（新版）
# =========================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

# 🔥 新模型名稱（注意要加 models/）
MODEL_NAME = "models/gemini-3.1-flash-lite-preview"

# =========================
# Prompt
# =========================
prompt = """
你是一個多模態事實分析系統。

請僅根據以下資訊判斷內容是否為假新聞，不可使用外部知識。

【影片主張】
{claim}

【影像描述】
{visual}

【整理知識】
{knowledge}

【外部證據（依可信度排序）】
{evidence}

【影像一致性】
{clip_desc}

0 = 真實內容
1 = 假新聞

最後請嚴格按照以下格式輸出（只能輸出兩行）：

LABEL: 0 或 LABEL: 1
REASON: 判斷依據
"""



# =========================
# 讀資料
# =========================
retrieve_data = read_jsonl_safe(input_path)
clip_data = read_jsonl_safe(clip_path)

clip_dict = {row["vid"]: row for row in clip_data}

# =========================
# 對齊資料
# =========================
aligned_data = []

for item in retrieve_data:
    vid = item["vid"]
    if vid in clip_dict:
        aligned_data.append(item)

aligned_data = aligned_data[:TOP_N]

# =========================
# 🔥 LLM 呼叫（新 + retry + 正確解析）
# =========================
def call_llm(prompt_text, retries=3):

    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config={"temperature": 0.2}
            )

            # 🔥 正確解析（關鍵）
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()

            elif hasattr(resp, "candidates"):
                parts = resp.candidates[0].content.parts
                return "".join([p.text for p in parts if hasattr(p, "text")]).strip()

            else:
                return str(resp)

        except Exception as e:
            print(f"[Retry {i+1}] {e}")

    return ""

# =========================
# worker（核心）
# =========================
def worker(item):

    vid = item["vid"]

    # ⭐ 直接用 merge 的欄位
    Rc = item.get("Rc", "")
    Rv = item.get("Rv", "")
    K  = item.get("K", "")

    evidence = item.get("evidence", [])
    label = item.get("label", -1)

    reason_text = ""
    pred_label = -1

    if not Rc:
        Rc = refine_dict.get(vid, {}).get("Rc", "")
    K = item.get("K", "")     # 原本的整理知識
    evidence = item.get("evidence", [])
    label = item.get("label", -1)

    clip_score = clip_dict.get(vid, {}).get("S", None)

    # =========================
    # CLIP 解讀（不要動）
    # =========================
    if clip_score is None:
        clip_desc = "影像一致性未知"
    elif clip_score > 0.5:
        clip_desc = "影像與文本高度一致"
    else:
        clip_desc = "影像與文本可能不一致"

    # =========================
    # evidence 過濾（不要動）
    # =========================
    clean_evidence = []

    for e in evidence:
        if isinstance(e, dict) and e.get("score", 0) > 2.5:
            clean_evidence.append(e)

    # 🔥 fallback（關鍵）
    if not clean_evidence:
        # 取前3筆當備用（避免完全沒證據）
        clean_evidence = evidence[:3]

    evidence = clean_evidence

    if not evidence:
        evidence_text = "無可靠外部證據"
    else:
        evidence_text = "\n".join(
            [f"{i+1}. {e.get('text','')} (score={e.get('score',0):.2f})"
             for i, e in enumerate(evidence[:5])]
        )

    # =========================
    # 🔥 LLM 推理（修正版）
    # =========================
    analysis = call_llm(
        prompt.format(
            claim=Rc,
            visual=Rv,      # ⭐ 新增
            knowledge=K,
            evidence=evidence_text,
            clip_desc=clip_desc
        )
    )
    pred_label = extract_label(analysis)
    reason_text = extract_reason(analysis)

    print(f"[DEBUG] {vid} → pred={pred_label} | reason={reason_text}")

    # 🔥 debug（升級）
    print(f"[DEBUG] {vid} → len={len(analysis)} | pred={pred_label}")


    return {
        "vid": vid,
        "analysis": analysis,
        "pred_label": pred_label,
        "reason": reason_text,   # ⭐ 新增
        "clip_score": clip_score,
        "evidence_used": [e.get("text", "") for e in evidence[:3]],
        "label": label
    }
def extract_label(text):
    if not text:
        return -1

    # 找 LABEL: 0 或 1（只抓數字）
    match = re.search(r'LABEL:\s*([01])', text)
    if match:
        return int(match.group(1))

    # fallback：直接找第一個 0/1
    match2 = re.search(r'\b([01])\b', text)
    if match2:
        return int(match2.group(1))

    return -1
def extract_reason(text):
    if not text:
        return ""

    match = re.search(r"REASON:\s*([\s\S]+)", text)
    if match:
        reason = match.group(1).strip()
        reason = reason.replace("\n", " ")
        return reason[:25]

    return ""
# =========================
# 多執行緒（🔥 改成2，避免爆API）
# =========================
results = []
lock = threading.Lock()

with ThreadPoolExecutor(max_workers=2) as executor:   # 🔥 不要改大

    futures = [executor.submit(worker, item) for item in aligned_data]

    for f in tqdm(as_completed(futures), total=len(futures)):
        res = f.result()

        with lock:
            results.append(res)

            with open(save_path, "a", encoding="utf-8") as out:
                out.write(json.dumps(res, ensure_ascii=False) + "\n")

print("Reason Done (Stable Gemini Version)")


# =========================
# 🎯 Evaluation + CSV輸出
# =========================
import csv

csv_path = os.path.join(output_dir, "result.csv")

correct = 0
total = 0

rows = []

for r in results:
    pred = r.get("pred_label", -1)
    gt = r.get("label", -1)

    # 只算有效的（避免 -1）
    if pred in [0, 1] and gt in [0, 1]:
        total += 1
        if pred == gt:
            correct += 1

    rows.append([r["vid"], pred, gt, r.get("reason", "")])

# 計算 accuracy
accuracy = correct / total if total > 0 else 0

# =========================
# 寫 CSV
# =========================
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # header
    writer.writerow(["vid", "pred_label", "label", "reason"])

    # data
    for row in rows:
        writer.writerow(row)

    # 最後一行
    writer.writerow(["ACCURACY", "", round(accuracy, 4)])

print(f"Saved CSV to: {csv_path}")
print(f"Accuracy: {accuracy:.4f}")