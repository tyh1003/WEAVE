import os
import json
import time
import re
import threading
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

INPUT_PATH = os.path.join("data", "ALL", "Final", "retrieve_new.jsonl")
OUTPUT_DIR = os.path.join("data", "ALL", "Final")
SAVE_PATH = os.path.join(OUTPUT_DIR, "reason_new.jsonl")
DEBUG_PATH = os.path.join(OUTPUT_DIR, "reason_new_debug.jsonl")
FAILED_PATH = os.path.join(OUTPUT_DIR, "reason_new_failed.jsonl")

REQUEST_TIMEOUT = 240
MODEL_NAME = "gemma-4-31b-it"
# MODEL_NAME = "gemini-3.1-flash-lite"

os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv()

# 讀取 API 金鑰
API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(10, 12)]
API_KEYS = [k for k in API_KEYS if k]


# Functions
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


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_retry_delay_429(error_msg):
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", error_msg)
    if match:
        return int(match.group(1))
    return 60


# ==================== 【修改處 1：更新 SYSTEM_PROMPT 納入 JSON 格式要求】 ====================
SYSTEM_PROMPT = """
你是一位專業的多模態事實查核專家，專精於辨識短影音中的假訊息、敘事操縱以及「脫離上下文」的誤導資訊。你的任務是嚴格根據提供的「背景知識」來驗證「文字輸入的核心內容」與「影像輸入的核心內容」描述的短影音內容之真實性。

在分析過程中，請特別警覺以下高階欺騙手段：
1. 脫離上下文：影像本身可能是真實的，但被賦予了錯誤的時間、地點或事件描述。
2. 地理政治重新映射：將事件發生的地點竄改為其他地區以達到特定目的。
3. 具體細節陷阱：在敘事中加入極其具體的細節（如精確的時間、人名）來增加可信度，但整體結論與事實不符。
4. 上下文劫持：利用真實的視覺證據來支持一個完全不相關且虛假的敘事結論。
5. 違背物理常識的內容：影像或文字描述中包含明顯違反物理定律或常識的元素。
6. 影像內容與文字描述的矛盾：影像中顯示的事件、人物或環境與文字描述的內容存在明顯不一致。

請採取以下思考步驟進行分析：
1. 文字分析：解析「文字輸入的核心內容」中的核心主張、時間、地點及關鍵事件。
2. 影像分析：分析「影像輸入的核心內容」的視覺描述，確認影像內容是否支持「文字輸入的核心內容」的主張。
3. 知識比對：將「文字輸入的核心內容」與「影像輸入的核心內容」共同與「背景知識」進行交叉比對，檢查是否存在矛盾或資訊缺失。
4. 綜合判定：基於上述比對，評估該短影音內容的真實程度。

請嚴格依照以下 JSON 格式回傳，確保鍵值（Keys）名稱完全一致，不要包含任何 Markdown 區塊標記（如 ```json）：
{
  "pred_label": 0,
  "conf": 0.95,
  "reason": "使用繁體中文完整且客觀說明判斷依據，不可輸出多餘描述。"
}

欄位輸出限制：
- pred_label: 短影音真偽判斷標籤。只能輸出 0 或 1 的整數。0 代表完全真實，1 代表存在誤導或矛盾。
- conf: 模型對此判斷的信心程度，輸出 0.00 至 1.00 之間的小數。越接近 1.00 代表信心越高。
- reason: 使用繁體中文提供簡潔且明確的判定理由。
- 所有輸出都以純文字輸出
- 所有輸出都必須以繁體中文輸出
"""

USER_PROMPT_TEMPLATE = """
你需要嚴格根據背景知識來判斷短影音真偽並輸出判斷依據。
- 文字輸入的核心內容：{Rc}
- 影像輸入的核心內容：{Rv}
- 背景知識：{K_int}{K_ext}
"""


# LLM
def count_input_tokens(api_key, contents, vid):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.count_tokens(model=MODEL_NAME, contents=contents)
        return int(response.total_tokens)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [❌ Token錯誤] [vid: {vid}] 計算失敗: {str(e)}")
        return -1

def generate_with_client(api_key, contents, vid):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] [🤖 LLM發送] [vid: {vid}] 開始建立 Fact-check Client 並發送推理判斷...")
    client = genai.Client(api_key=api_key)

    try:
        # ==================== 【修改處 2：移除 response_schema 設定】 ====================
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json", # 保留此行以穩定 JSON 格式
            )
        )

        print(f"[{time.strftime('%H:%M:%S')}] [✅ LLM接收] [vid: {vid}] 推理判定完成！耗時: {round(time.time()-t_start, 2)} 秒")

        # ----------------- 防禦核心 1：安全獲取文字 -----------------
        raw_text = ""
        try:
            if hasattr(response, "text") and response.text:
                raw_text = response.text.strip()
            else:
                raw_text = response.candidates[0].content.parts[0].text.strip()
        except Exception as text_err:
            print(f"[{time.strftime('%H:%M:%S')}] [⚠️ 提取警告] [vid: {vid}] 無法直接讀取 response.text: {str(text_err)}")
            raw_text = str(response).strip()

        # ----------------- 防禦核心 2：清洗 Markdown 標籤 -----------------
        if "```" in raw_text:
            raw_text = re.sub(r'\s*```$', '', raw_text).strip()

        # ----------------- 防禦核心 3：安全解析 JSON -----------------
        data = {}
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as je:
            print(f"[{time.strftime('%H:%M:%S')}] [🚨 JSON解析失敗] [vid: {vid}] 偵測到非法格式，啟動修復機制。錯誤: {str(je)}")
            print(f"\n========== RAW RESPONSE ==========")
            print(repr(raw_text))
            print(f"========== END RESPONSE ==========\n")            
            
            try:
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_json = raw_text[start_idx:end_idx + 1]
                    data = json.loads(cleaned_json)
                    print(f"[{time.strftime('%H:%M:%S')}] [✨ 修復成功] [vid: {vid}] 透過 JSON 區間裁剪成功解析資料！")
                else:
                    raise ValueError("找不到完整的 JSON 大括號結構")
            except Exception:
                try:
                    first_line = raw_text.split('\n')[0].strip()
                    data = json.loads(first_line)
                    print(f"[{time.strftime('%H:%M:%S')}] [✨ 修復成功] [vid: {vid}] 透過抓取首行 JSON 成功解析資料！")
                except Exception as final_err:
                    print(f"[{time.strftime('%H:%M:%S')}] [💥 修復絕望] [vid: {vid}] 所有修復手段均失效。")
                    raise final_err

        # ----------------- 欄位解析與數值安全防禦 -----------------
        try:
            pred_label = int(data.get("pred_label", 0))
            if pred_label not in [0, 1]:
                pred_label = 1 if pred_label > 0.5 else 0
        except:
            pred_label = 0

        try:
            conf = round(float(data.get("conf", 0.5)), 4)
            conf = max(0.0, min(1.0, conf))
        except:
            conf = 0.5

        result = {
            "pred_label": pred_label,
            "conf": conf,
            "reason": str(data.get("reason", "")).strip()
        }
        return result, raw_text

    except Exception as e:
        error_msg = str(e)
        print(f"[{time.strftime('%H:%M:%S')}] [❌ API異常] [vid: {vid}] 發生錯誤: {error_msg}")

        if "429" in error_msg:
            delay = extract_retry_delay_429(error_msg)
            print(f"[{time.strftime('%H:%M:%S')}] [⏳ 速率限制] [vid: {vid}] 觸發 429，執行緒睡眠 {delay} 秒...")
            time.sleep(delay)

        return None, error_msg


def generate_with_timeout(api_key, contents, vid, timeout=120):
    print(f"[{time.strftime('%H:%M:%S')}] [⚡ 核心防護] [vid: {vid}] 設定超時監控 ({timeout}秒)")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"TimeoutMonitor-{vid}")
    future = executor.submit(generate_with_client, api_key, contents, vid)

    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except TimeoutError:
        print(f"[{time.strftime('%H:%M:%S')}] [⏰ Timeout] [vid: {vid}] 超過 {timeout} 秒")
        future.cancel()
        executor.shutdown(wait=False)
        print(f"[{time.strftime('%H:%M:%S')}] [🧵 Active Threads] {threading.active_count()}")
        return None, f"TIMEOUT_AFTER_{timeout}s"
    except Exception as e:
        executor.shutdown(wait=False)
        print(f"[{time.strftime('%H:%M:%S')}] [🚨 超時/中斷] [vid: {vid}] {repr(e)}")
        return None, repr(e)


# 全域鎖定與狀態集合初始化
lock = threading.Lock()
processed_vids = set()
in_progress_vids = set()


# Main process row
def process_row(args):
    global processed_vids, in_progress_vids
    start_time = time.time()
    row, api_key = args
    vid = str(row["vid"])
    key_hint = f"...{api_key[-6:]}" if api_key else "None"

    print(f"[{time.strftime('%H:%M:%S')}] [📥 任務喚醒] [vid: {vid}] 使用金鑰 {key_hint} 進入查核流程")

    with lock:
        if (vid in processed_vids or vid in in_progress_vids):
            print(f"[{time.strftime('%H:%M:%S')}] [⏭️ 跳過] [vid: {vid}] 已在名單中。")
            return None
        in_progress_vids.add(vid)

    try:
        Rc = str(row.get("Rc", "")).strip()
        Rv = str(row.get("Rv", "")).strip()
        K_int = str(row.get("K_int", "")).strip()
        K_ext = str(row.get("K_ext", "")).strip()

        prompt = USER_PROMPT_TEMPLATE.format(Rc=Rc, Rv=Rv, K_int=K_int, K_ext=K_ext)
        contents = [prompt]

        print(f"[{time.strftime('%H:%M:%S')}] [📊 Token計算] [vid: {vid}] 正在向 API 查詢本次 RAG 輸入的 Token 總數...")
        input_tokens = count_input_tokens(api_key, contents, vid)
        print(f"[{time.strftime('%H:%M:%S')}] [📊 Token計算] [vid: {vid}] 計算完成。Token 數: {input_tokens}")

        result, raw_text = generate_with_timeout(api_key, contents, vid, timeout=REQUEST_TIMEOUT)

        if result is None:
            print(f"[{time.strftime('%H:%M:%S')}] [💾 寫入錯誤檔] [vid: {vid}] 查核失敗，準備獲取鎖定寫入 Failed Log")
            with lock:
                append_jsonl(FAILED_PATH, {"vid": vid, "token": input_tokens, "reason": raw_text})
            return None

        elapsed_time = round(time.time() - start_time, 4)
        save_obj = {
            "vid": vid,
            "label": int(row["label"]),
            "token": input_tokens,
            "time": elapsed_time,
            "pred_label": result["pred_label"],
            "conf": result["conf"],
            "reason": result["reason"]
        }

        debug_obj = {
            "vid": vid,
            "label": int(row["label"]),
            "token": input_tokens,
            "time": elapsed_time,
            "Rc": Rc,
            "Rv": Rv,
            "K_int": K_int,
            "K_ext": K_ext,
            "prompt": prompt,
            "raw_response": raw_text
        }

        print(f"[{time.strftime('%H:%M:%S')}] [💾 準備存檔] [vid: {vid}] 判定成功，等待 I/O 鎖定進行寫入...")
        with lock:
            append_jsonl(SAVE_PATH, save_obj)
            append_jsonl(DEBUG_PATH, debug_obj)
            processed_vids.add(vid)

        print(f"[{time.strftime('%H:%M:%S')}] [✨ 存檔成功] [vid: {vid}] 事實查核結果已寫入檔案。")
        return vid

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [💥 執行緒崩難] [vid: {vid}] 拋出未捕捉異常: {str(e)}")
        with lock:
            append_jsonl(FAILED_PATH, {"vid": vid, "reason": str(e)})
        return None
    finally:
        with lock:
            in_progress_vids.discard(vid)


# ---------------- Windows 執行核心保護 ----------------
if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] [🚀 系統啟動] 推理判定模組初始化開始...")

    if not API_KEYS:
        print(f"[{time.strftime('%H:%M:%S')}] [❌ 嚴重錯誤] 找不到 GEMINI_API_KEY_11！")
        import sys
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] [🔑 金鑰就緒] 成功載入 {len(API_KEYS)} 個推理專用 API Key。")

    save_df = read_jsonl_keep_vid_as_str(SAVE_PATH)
    if not save_df.empty:
        processed_vids = set(save_df["vid"].astype(str))

    print(f"[{time.strftime('%H:%M:%S')}] [📦 進度載入] 歷史查核進度載入成功，已跳過 {len(processed_vids)} 筆已完成項目。")

    print(f"[{time.strftime('%H:%M:%S')}] [📖 檔案讀取] 正在讀取 retrieve.jsonl...")
    df = read_jsonl_keep_vid_as_str(INPUT_PATH)
    if df.empty:
        raise RuntimeError(f"錯誤：輸入檔案 {INPUT_PATH} 為空或不存在")

    df["vid"] = df["vid"].astype(str)
    df = df[~df["vid"].isin(processed_vids)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [🎉 完工] 所有資料皆已完成事實查核推理！")
        import sys
        sys.exit(0)

    tasks = [
        (row, API_KEYS[i % len(API_KEYS)])
        for i, (_, row) in enumerate(df.iterrows())
    ]

    print(f"[{time.strftime('%H:%M:%S')}] [🔥 執行緒準備] 剩餘查核任務總數: {len(tasks)}，平行執行緒上限: {len(API_KEYS)}")

    success = 0
    with ThreadPoolExecutor(max_workers=len(API_KEYS), thread_name_prefix="ReasonWorker") as executor:
        futures = [executor.submit(process_row, t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Reasoning"):
            try:
                result = f.result()
                if result is not None:
                    success += 1
            except Exception as e:
                print(f"\n[{time.strftime('%H:%M:%S')}] [🚨 主池回報異常] 收集 Future 結果發生錯誤: {str(e)}")

    print(f"\n[{time.strftime('%H:%M:%S')}] [🏁 腳本終點] 查核判定完畢！本次成功寫入: {success} 筆。")