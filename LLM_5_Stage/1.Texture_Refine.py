import os
import json
import time
import re
import threading
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from multiprocessing import (
    Process,
    Queue
)

from pydantic import BaseModel

from google import genai
from google.genai import types


INPUT_PATH = "data/All/all.jsonl"

OUTPUT_DIR = "data/All/refine/"

REQUEST_TIMEOUT = 180

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "Rc.jsonl"
)

DEBUG_PATH = os.path.join(
    OUTPUT_DIR,
    "Rc_debug.jsonl"
)

FAILED_PATH = os.path.join(
    OUTPUT_DIR,
    "Rc_failed.jsonl"
)

MODEL_NAME = "gemma-4-31b-it"

load_dotenv()

API_KEYS = [
    # os.getenv("GEMINI_API_KEY_1"),
    # os.getenv("GEMINI_API_KEY_2"),
    # os.getenv("GEMINI_API_KEY_3"),
    # os.getenv("GEMINI_API_KEY_4"),
    # os.getenv("GEMINI_API_KEY_5"),
    # os.getenv("GEMINI_API_KEY_6"),
    # os.getenv("GEMINI_API_KEY_7"),
    # os.getenv("GEMINI_API_KEY_8"),
    # os.getenv("GEMINI_API_KEY_9"),
    os.getenv("GEMINI_API_KEY_10"),
    os.getenv("GEMINI_API_KEY_11"),
]

API_KEYS = [
    k for k in API_KEYS
    if k
]

if len(API_KEYS) == 0:

    raise RuntimeError(
        "No API keys found"
    )


class RcResponse(BaseModel):

    Rc: str


def read_jsonl_keep_vid_as_str(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

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

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False
            ) + "\n"
        )


def extract_retry_delay_429(error_msg):

    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s",
        error_msg
    )

    if match:
        return int(match.group(1))

    return 60


def extract_json_from_text(text):

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```json",
        "",
        text
    )

    text = re.sub(
        r"^```",
        "",
        text
    )

    text = re.sub(
        r"```$",
        "",
        text
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return ""

    try:

        obj = json.loads(
            match.group(0)
        )

        return obj.get(
            "Rc",
            ""
        ).strip()

    except:

        return ""


SYSTEM_PROMPT = """
你是一個多模態假訊息分析系統，需要完成任務並嚴格的遵守限制。

任務：
- 精煉輸入的雜訊文本，並抽取文本核心主張

限制：
- 不可摘要整段內容
- 不可推測或補充不存在的內容
- 輸出必須嚴格符合 JSON schema
- 使用繁體中文回答
"""


USER_PROMPT_TEMPLATE = """
文本資訊：

- title:
{title}

- ocr:
{ocr}

- transcript:
{transcript}
"""


def generate_with_client(
    api_key,
    contents
):

    client = genai.Client(
        api_key=api_key
    )

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=contents,

            config=types.GenerateContentConfig(

                temperature=0.2,

                response_mime_type="application/json",

                response_schema=RcResponse,

                system_instruction=SYSTEM_PROMPT
            )
        )

        raw_text = ""

        try:

            raw_text = response.text

        except:

            raw_text = str(response)

        parsed = response.parsed

        # 正常 parsed
        if parsed:

            result = {
                "Rc": parsed.Rc.strip()
            }

            return result, raw_text

        # fallback parser
        Rc = extract_json_from_text(
            raw_text
        )

        if Rc != "":

            result = {
                "Rc": Rc
            }

            return result, raw_text

        return None, raw_text

    except Exception as e:

        error_msg = str(e)

        # exception 裡面也可能藏 JSON
        Rc = extract_json_from_text(
            error_msg
        )

        if Rc != "":

            result = {
                "Rc": Rc
            }

            return result, error_msg

        if "429" in error_msg:

            delay = extract_retry_delay_429(
                error_msg
            )

            time.sleep(delay)

        return None, error_msg


def worker_generate(
    queue,
    api_key,
    contents
):

    result = generate_with_client(
        api_key,
        contents
    )

    queue.put(result)


def generate_with_timeout(
    api_key,
    contents,
    timeout=120
):

    queue = Queue()

    process = Process(

        target=worker_generate,

        args=(
            queue,
            api_key,
            contents
        )
    )

    process.start()

    process.join(timeout)

    # timeout
    if process.is_alive():

        process.terminate()

        process.join()

        return (
            None,
            f"TIMEOUT_AFTER_{timeout}_SECONDS"
        )

    # 正常完成
    if not queue.empty():

        return queue.get()

    return (
        None,
        "EMPTY_PROCESS_RESULT"
    )


save_df = read_jsonl_keep_vid_as_str(
    SAVE_PATH
)

processed_vids = (

    set(save_df["vid"].astype(str))

    if not save_df.empty

    else set()
)

df = read_jsonl_keep_vid_as_str(
    INPUT_PATH
)

if df.empty:

    raise RuntimeError(
        f"{INPUT_PATH} empty"
    )

df["vid"] = df["vid"].astype(str)

df = df[df["label"] != 2]

df = df.drop_duplicates(
    subset=["vid"]
)

df = df[
    ~df["vid"].isin(processed_vids)
]

df = df.reset_index(drop=True)

lock = threading.Lock()

in_progress_vids = set()


def process_row(args):

    row, api_key = args

    vid = str(row["vid"])

    with lock:

        if (
            vid in processed_vids
            or vid in in_progress_vids
        ):

            return None

        in_progress_vids.add(vid)

    try:

        title = str(
            row.get("title", "")
        ).strip()

        ocr = str(
            row.get("ocr", "")
        ).strip()

        transcript = str(
            row.get("transcript", "")
        ).strip()

        if (
            title == ""
            and ocr == ""
            and transcript == ""
        ):

            with lock:

                append_jsonl(

                    FAILED_PATH,

                    {
                        "vid": vid,
                        "reason": "empty_text_input"
                    }
                )

            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            ocr=ocr,
            transcript=transcript
        )

        result, raw_text = generate_with_timeout(
            api_key,
            prompt,
            timeout=REQUEST_TIMEOUT
        )

        if result is None:

            with lock:

                append_jsonl(

                    FAILED_PATH,

                    {
                        "vid": vid,
                        "reason": raw_text
                    }
                )

            return None

        save_obj = {

            "vid": vid,

            "label": int(row["label"]),

            **result
        }

        debug_obj = {

            "vid": vid,

            "prompt": prompt,

            "raw_response": raw_text
        }

        with lock:

            append_jsonl(
                SAVE_PATH,
                save_obj
            )

            append_jsonl(
                DEBUG_PATH,
                debug_obj
            )

            processed_vids.add(vid)

        return vid

    except Exception as e:

        with lock:

            append_jsonl(

                FAILED_PATH,

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

    (
        row,
        API_KEYS[i % len(API_KEYS)]
    )

    for i, (_, row)
    in enumerate(df.iterrows())
]

success = 0

with ThreadPoolExecutor(
    max_workers=len(API_KEYS)
) as executor:

    futures = [

        executor.submit(
            process_row,
            t
        )

        for t in tasks
    ]

    for f in tqdm(
        as_completed(futures),
        total=len(futures)
    ):

        try:

            result = f.result()

            if result is not None:
                success += 1

        except Exception as e:

            print(
                f"Future Error: {str(e)}"
            )


print(f"Success: {success}")

print(
    "Step1.1 Text Refine Done."
)