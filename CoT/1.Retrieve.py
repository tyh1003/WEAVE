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

INPUT_PATH = os.path.join("data", "ALL", "all.jsonl")
IMAGE_ROOT = os.path.join("data", "ALL", "frames_16")
OUTPUT_DIR = os.path.join("data", "ALL", "Final_v3")
SAVE_PATH = os.path.join(OUTPUT_DIR, "retrieve_new.jsonl")
DEBUG_PATH = os.path.join(OUTPUT_DIR, "retrieve_new_debug.jsonl")
FAILED_PATH = os.path.join(OUTPUT_DIR, "retrieve_new_failed.jsonl")

REQUEST_TIMEOUT = 1500
MODEL_NAME = "gemma-4-31b-it"
# MODEL_NAME = "gemini-3.1-flash-lite"

os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv()

# 讀取 API 金鑰
API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 12)]
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


def load_image_bytes(path):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        data = f.read()
        if not data:
            print(f"[{time.strftime('%H:%M:%S')}] [⚠️ 警告] [Empty Image] {path}")
            return None
        return data


def extract_grounding_metadata(response):
    queries = []
    uris = []
    try:
        grounding = response.candidates[0].grounding_metadata
        if hasattr(grounding, "web_search_queries"):
            queries = grounding.web_search_queries or []

        if hasattr(grounding, "grounding_chunks"):
            for chunk in grounding.grounding_chunks:
                web_info = getattr(chunk, "web", None)
                if not web_info:
                    continue
                if getattr(web_info, "uri", None):
                    uris.append(web_info.uri)
    except:
        pass

    return {
        "query": list(dict.fromkeys(queries)),
        "uri": list(dict.fromkeys(uris))
    }


# ==================== 【修改處 1：更新 SYSTEM_PROMPT 納入 JSON 格式要求】 ====================
SYSTEM_PROMPT = """
你是一個多模態訊息分析與事實查核系統。你必須依序完成以下的任務，並且嚴格的遵守每個任務的限制與共同限制。

共同限制：
- 必須保持客觀
- 不可推測與幻想不存在的內容
- 所有輸出都以純文字輸出
- 所有輸出都必須以繁體中文輸出
- 依照指定的 JSON 格式輸出，不要包含任何 Markdown 區塊標記（如 ```json）

任務一(Rc)：整理輸入的文本資訊，提取其核心內容並保留關鍵細節，且須移除文本中的雜訊。
任務一限制：
- 只能使用輸入之文本資訊，不可參考其他任何資訊
- 不可摘要整段內容
- 需保留文本中的關鍵細節（如人物、事件、時間、地點等）
- 時間須避免今日、昨日等相對時間的描述，使用準確日期或時間點

任務二(Rv)：考慮時間序列來描述影像中的視覺內容。
任務二限制：
- 只能使用影像資訊，不可參考其他任何資訊
- 僅描述可直接觀察到的內容（人物、物件、動作、場景）
- 必須考慮時間序列，描述事件的發展過程，而非單一靜態畫面
- 避免每張影像獨立描述，應綜合多張影像來理解事件的演變

任務三(K_int):根據 Rc 與 Rv 的內容，輸出能用於事實查核之相關背景知識。
任務三限制：
- 僅根據你的內部知識
- 不要出現任何說明性語句，直接給出與內容相關的背景知識
- 內容應具權威性，避免不可靠或爭議性的資訊

任務四(K_ext):根據 Rc 與 Rv 的內容，使用 Google Search tool 工具搜尋並整理能用於事實查核之相關背景知識。
任務四限制：
- 必須根據 Rc 和 Rv 的內容來決定搜尋的關鍵字，禁止無差別搜尋
- 必須使用 Google Search tool 工具進行搜尋，禁止直接使用內部知識
- 不要出現任何說明性語句，直接給出與內容相關的背景知識
- 搜尋結果須與 Rc 和 Rv 的內容相關，著重於該次事件、人物、地點、內容等的相關背景知識
- 只整理與事件相關且具權威性的資訊，避免無關或不可靠的內容

請嚴格依照以下 JSON 格式回傳，確保鍵值（Keys）名稱完全一致：
{
  "Rc": "任務一的繁體中文輸出內容...",
  "Rv": "任務二的繁體中文輸出內容...",
  "K_int": "任務三的繁體中文輸出內容...",
  "K_ext": "任務四的繁體中文輸出內容..."
}
"""

USER_PROMPT_TEMPLATE = """
使用前面輸入的影像順序與下方的文本資訊，嚴格遵守每個任務的限制與共同限制，並依序完成所有任務。

文本資訊：
- 標題:{title}
- 影像內文字：{ocr}
- 語音轉錄:{transcript}

影像與時間軸說明：
- 前方附加的影像中，第一張影像為最早的時間點，最後一張影像為最晚時間點
- 請根據時間順序理解事件發展
"""


# LLM
def count_input_tokens(api_key, contents):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.count_tokens(model=MODEL_NAME, contents=contents)
        return int(response.total_tokens)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [❌ 錯誤] Token 計算失敗: {str(e)}")
        return -1


def generate_with_client(api_key, contents, vid):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] [🤖 LLM發送] [vid: {vid}] 開始建立 Client 並發送 generate_content...")
    client = genai.Client(api_key=api_key)

    try:
        # ==================== 【修改處 2：移除 response_schema 設定】 ====================
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                tools=[{"google_search": {}}],
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json"
            )
        )

        print(f"[{time.strftime('%H:%M:%S')}] [✅ LLM接收] [vid: {vid}] 成功取得回應！耗時: {round(time.time()-t_start, 2)} 秒")

        raw_text = ""
        try:
            raw_text = response.text
        except:
            raw_text = str(response)

        data = json.loads(raw_text.strip())
        grounding_info = extract_grounding_metadata(response)

        result = {
            "Rc": data.get("Rc", "").strip(),
            "Rv": data.get("Rv", "").strip(),
            "K_int": str(data.get("K_int", "")).strip(),
            "K_ext": str(data.get("K_ext", "")).strip(),
            "query": grounding_info.get("query", []),
            "uri": grounding_info.get("uri", [])
        }
        return result, raw_text

    except Exception as e:
        error_msg = str(e)
        print(f"[{time.strftime('%H:%M:%S')}] [❌ API異常] [vid: {vid}] 發生錯誤: {error_msg}")

        if "429" in error_msg:
            delay = extract_retry_delay_429(error_msg)
            print(f"[{time.strftime('%H:%M:%S')}] [⏳ 速率限制] [vid: {vid}] 觸發 429，執行緒將睡眠 {delay} 秒...")
            time.sleep(delay)

        return None, error_msg


def generate_with_timeout(api_key, contents, vid, timeout=120):
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
        print(f"[{time.strftime('%H:%M:%S')}] [🚨 子執行緒異常] [vid: {vid}] {repr(e)}")
        return None, repr(e)


# 全域鎖定與狀態追蹤
lock = threading.Lock()
processed_vids = set()
in_progress_vids = set()


def process_row(args):
    global processed_vids, in_progress_vids
    start_time = time.time()
    row, api_key = args
    vid = str(row["vid"])
    key_hint = f"...{api_key[-6:]}" if api_key else "None"

    print(f"[{time.strftime('%H:%M:%S')}] [📥 任務喚醒] [vid: {vid}] 使用金鑰 {key_hint} 開始進入工作排程")

    with lock:
        if (vid in processed_vids or vid in in_progress_vids):
            return None
        in_progress_vids.add(vid)

    try:
        title = str(row.get("title", "")).strip()
        ocr = str(row.get("ocr", "")).strip()
        transcript = str(row.get("transcript", "")).strip()

        prompt = USER_PROMPT_TEMPLATE.format(title=title, ocr=ocr, transcript=transcript)
        contents = []

        image_paths = [os.path.join(IMAGE_ROOT, vid, f"frame_{i}.jpg") for i in range(1, 17)]
        for p in image_paths:
            img_bytes = load_image_bytes(p)
            if img_bytes is None:
                continue
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        contents.append(prompt)
        input_tokens = count_input_tokens(api_key, contents)
        result, raw_text = generate_with_timeout(api_key, contents, vid, timeout=REQUEST_TIMEOUT)

        if result is None:
            print(f"[{time.strftime('%H:%M:%S')}] [💾 寫入錯誤檔] [vid: {vid}] API 回傳失敗，將寫入 Failed Log")
            with lock:
                append_jsonl(FAILED_PATH, {"vid": vid, "token": input_tokens, "reason": raw_text})
            return None

        elapsed_time = round(time.time() - start_time, 4)
        save_obj = {
            "vid": vid,
            "label": int(row["label"]),
            "token": input_tokens,
            "time": elapsed_time,
            **result
        }

        debug_obj = {
            "vid": vid,
            "token": input_tokens,
            "prompt": prompt,
            "image_paths": image_paths,
            "raw_response": raw_text,
            "grounding": {
                "query": result.get("query", []),
                "uri": result.get("uri", [])
            }
        }

        print(f"[{time.strftime('%H:%M:%S')}] [💾 準備存檔] [vid: {vid}] 處理完畢，正在等待 I/O 鎖定進行寫入...")
        with lock:
            append_jsonl(SAVE_PATH, save_obj)
            append_jsonl(DEBUG_PATH, debug_obj)
            processed_vids.add(vid)

        print(f"[{time.strftime('%H:%M:%S')}] [✨ 存檔成功] [vid: {vid}] 資料已成功永久化儲存。")
        return vid

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [💥 執行緒崩潰] [vid: {vid}] 拋出未捕捉異常: {str(e)}")
        with lock:
            append_jsonl(FAILED_PATH, {"vid": vid, "reason": str(e)})
        return None
    finally:
        with lock:
            in_progress_vids.discard(vid)


# ---------------- Windows 執行核心保護 ----------------
if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] [🚀 系統啟動] 正在初始化環境變數與檔案進度...")

    if not API_KEYS:
        print(f"[{time.strftime('%H:%M:%S')}] [❌ 嚴重錯誤] 找不到任何合法的 GEMINI_API_KEY_X！")
        import sys
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] [🔑 金鑰就緒] 成功載入 {len(API_KEYS)} 個 API Key。")

    save_df = read_jsonl_keep_vid_as_str(SAVE_PATH)
    if not save_df.empty:
        processed_vids = set(save_df["vid"].astype(str))

    print(f"[{time.strftime('%H:%M:%S')}] [📦 進度載入] 已跳過 {len(processed_vids)} 筆已處理項目。")

    df = read_jsonl_keep_vid_as_str(INPUT_PATH)
    if df.empty:
        raise RuntimeError(f"{INPUT_PATH} is empty or does not exist.")

    df["vid"] = df["vid"].astype(str)
    df = df[df["label"] != 2]
    df = df[~df["vid"].isin(processed_vids)]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [🎉 完工] 所有資料皆已處理完畢！")
        import sys
        sys.exit(0)

    tasks = [
        (row, API_KEYS[i % len(API_KEYS)])
        for i, (_, row) in enumerate(df.iterrows())
    ]

    print(f"[{time.strftime('%H:%M:%S')}] [🔥 執行緒準備] 剩餘總工作數: {len(tasks)}，平行執行緒上限: {len(API_KEYS)}")
    success = 0

    with ThreadPoolExecutor(max_workers=len(API_KEYS), thread_name_prefix="MainWorker") as executor:
        futures = [executor.submit(process_row, t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = f.result()
                if result is not None:
                    success += 1
            except Exception as e:
                print(f"\n[{time.strftime('%H:%M:%S')}] [🚨 主池回報異常] Future 收集發生錯誤: {str(e)}")

    print(f"\n[{time.strftime('%H:%M:%S')}] [🏁 腳本終點] 執行完畢！本次成功處理: {success} 筆資料。")