import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

from google import genai
from google.genai import types


CONFIG = {
    "dataset": "All",
    "model_name": "gemma-4-31b-it",
}

dataset = CONFIG["dataset"]
MODEL_NAME = CONFIG["model_name"]

output_dir = f"data/All/retrieve/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Kint.jsonl")
debug_path = os.path.join(output_dir, "Kint_debug.jsonl")
failed_path = os.path.join(output_dir, "Kint_failed.jsonl")

input_path = f"data/All/refine/refine.jsonl"

load_dotenv()

API_KEYS = [
    # os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9")
]

API_KEYS = [k for k in API_KEYS if k]

if len(API_KEYS) == 0:
    raise RuntimeError("No API keys found")

clients = [genai.Client(api_key=k) for k in API_KEYS]


class KintResponse(BaseModel):
    K_int: str


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


def extract_json_from_text(text):
    if not text:
        return ""

    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return ""

    json_str = match.group(0)

    try:
        obj = json.loads(json_str)
        return obj.get("K_int", "").strip()
    except:
        return ""


def generate_with_client(client, content):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=KintResponse,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high"
                ),
                system_instruction=(
                    "你是一個知識檢索模組。"
                    "你的任務是輸出純背景知識。"
                    "輸出格式必須為 JSON：{\"K_int\": \"內容\"}。"
                    "內容需為單一段落、繁體中文、不可包含說明。"
                )
            )
        )

        raw_text = ""

        try:
            raw_text = response.text
        except:
            raw_text = str(response)

        if response and response.parsed:
            return response.parsed.K_int.strip(), raw_text

        K_int = extract_json_from_text(raw_text)

        if K_int != "":
            return K_int, raw_text

        return "", raw_text

    except Exception as e:
        error_msg = str(e)

        if "429" in error_msg:
            delay = extract_retry_delay_429(error_msg)
            time.sleep(delay)

        return "", error_msg


def build_prompt(Rc, Rv):
    return f"""
你是一個知識檢索模組，請輸出「純背景知識」。

輸入：
文本主張：
{Rc}

視覺描述：
{Rv}

輸出要求（嚴格遵守）：
1. 只輸出客觀背景知識
2. 不要出現任何說明性語句
3. 不要使用條列、標題、markdown
4. 不要輸出換行符號（\\n）
5. 請輸出為單一段落
6. 內容應為百科式敘述
7. 使用繁體中文
"""


if not os.path.exists(input_path):
    raise RuntimeError(f"{input_path} not found")


save_df = read_jsonl_keep_vid_as_str(output_path)

processed_vids = (
    set(save_df["vid"].astype(str))
    if not save_df.empty
    else set()
)

df = read_jsonl_keep_vid_as_str(input_path)

if df.empty:
    raise RuntimeError(f"{input_path} is empty")


df["vid"] = df["vid"].astype(str)

df = df[~df["vid"].isin(processed_vids)]
df = df.reset_index(drop=True)


lock = threading.Lock()
in_progress_vids = set()


def process_row(args):
    row, client = args

    vid = str(row["vid"])

    with lock:
        if vid in processed_vids or vid in in_progress_vids:
            return None

        in_progress_vids.add(vid)

    try:
        Rc = row.get("Rc", "")
        Rv = row.get("Rv", "")
        label = int(row.get("label", -1))

        if Rc == "" and Rv == "":
            with lock:
                append_jsonl(
                    failed_path,
                    {
                        "vid": vid,
                        "reason": "empty_input"
                    }
                )

            return None

        prompt = build_prompt(Rc, Rv)
        
        K_int, error_msg = generate_with_client(
            client,
            prompt
        )
    

        K_int = K_int.strip()

        if K_int == "":
            with lock:
                append_jsonl(
                    failed_path,
                    {
                        "vid": vid,
                        "reason": error_msg if error_msg else "empty_llm_response"
                    }
                )

            return None

        result = {
            "vid": vid,
            "K_int": K_int,
            "label": label
        }

        debug = {
            "vid": vid,
            "label": label,
            "Rc": Rc,
            "Rv": Rv,
            "llm_prompt": prompt,
            "llm_output_raw": K_int
        }

        with lock:
            append_jsonl(output_path, result)
            append_jsonl(debug_path, debug)

            processed_vids.add(vid)

        return vid

    except Exception as e:
        with lock:
            append_jsonl(
                failed_path,
                {
                    "vid": vid,
                    "reason": str(e)
                }
            )

        return None

    finally:
        with lock:
            in_progress_vids.discard(vid)


tasks = [
    (row, clients[i % len(clients)])
    for i, (_, row) in enumerate(df.iterrows())
]

success = 0

with ThreadPoolExecutor(max_workers=len(clients)) as executor:
    futures = [
        executor.submit(process_row, t)
        for t in tasks
    ]

    for f in tqdm(as_completed(futures), total=len(futures)):
        result = f.result()

        if result is not None:
            success += 1


print(f"Success: {success}")
print("Step2.1 LLM Retrieve Done.")
