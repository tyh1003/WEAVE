from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json
import time
import re
import threading
import argparse

from dotenv import load_dotenv
from pydantic import BaseModel
from multiprocessing import (
    Process,
    Queue
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from google import genai
from google.genai import types


# =========================================================
# Argument
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data',
    type=str,
    default='ALL'
)

args = parser.parse_args()

dataset = args.data


# =========================================================
# Config
# =========================================================
MODEL_NAME = "gemma-4-31b-it"

INPUT_PATH = "data/ALL/all.jsonl"

OUTPUT_DIR = f"data/ALL/CoT/gemma-4-31b-it/"

TEXT_REFINE_PATH = os.path.join(
    OUTPUT_DIR,
    "lm_text_refine.jsonl"
)

VISION_REFINE_PATH = os.path.join(
    OUTPUT_DIR,
    "lm_visual_refine_quad4.jsonl"
)

SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "lm_retrieve_quad4.jsonl"
)

DEBUG_PATH = os.path.join(
    OUTPUT_DIR,
    "lm_retrieve_quad4_debug.jsonl"
)

FAILED_PATH = os.path.join(
    OUTPUT_DIR,
    "lm_retrieve_quad4_failed.jsonl"
)

REQUEST_TIMEOUT = 240

os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv(override=True)

API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}")
    for i in range(10, 12)
]

API_KEYS = [
    k for k in API_KEYS
    if k
]

if len(API_KEYS) == 0:
    raise RuntimeError("No Gemini API Keys Found")


# =========================================================
# Prompt
# =========================================================
PROMPT_TEMPLATE = """
Analyze the {text} from both textual and visual descriptions of the micro-video. Use your pre-trained knowledge to provide relevant background information, enhancing comprehension of the video context, without assessing authenticity. Ensure responses are concise, and focused.
"""


# =========================================================
# Response Schema
# =========================================================
class RetrieveResponse(BaseModel):
    refined_text: str


# =========================================================
# Utils
# =========================================================
def append_jsonl(path, obj):

    with open(path, "a", encoding="utf-8") as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False
            ) + "\n"
        )


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


def extract_retry_delay_429(error_msg):

    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s",
        error_msg
    )

    if match:
        return int(match.group(1))

    return 60


# =========================================================
# Parsing Function
# =========================================================
def clean_json_text(raw_text):

    if raw_text is None:
        return ""

    text = str(raw_text).strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    return text


def parse_response(raw_text):

    try:

        cleaned = clean_json_text(raw_text)

        obj = json.loads(cleaned)

        if isinstance(obj, dict):

            refined_text = obj.get(
                "refined_text",
                ""
            )

            if refined_text is not None:

                return {
                    "refined_text": str(refined_text).strip()
                }

    except:
        pass

    return None


# =========================================================
# Gemini
# =========================================================
def count_input_tokens(api_key, contents):

    try:

        client = genai.Client(api_key=api_key)

        response = client.models.count_tokens(
            model=MODEL_NAME,
            contents=contents
        )

        return int(response.total_tokens)

    except:
        return -1


def generate_with_client(api_key, contents):

    client = genai.Client(api_key=api_key)

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=RetrieveResponse
            )
        )

        raw_text = ""

        try:
            raw_text = response.text
        except:
            raw_text = str(response)

        # =================================================
        # First: official parsed
        # =================================================
        parsed = None

        try:
            parsed = response.parsed
        except:
            parsed = None

        if parsed:

            result = {
                "refined_text": parsed.refined_text.strip()
            }

            return result, raw_text

        # =================================================
        # Second: manual parsing fallback
        # =================================================
        parsed_result = parse_response(raw_text)

        if parsed_result is not None:
            return parsed_result, raw_text

        return None, raw_text

    except Exception as e:

        error_msg = str(e)

        # =================================================
        # Sometimes exception contains json
        # =================================================
        parsed_result = parse_response(error_msg)

        if parsed_result is not None:
            return parsed_result, error_msg

        if "429" in error_msg:

            delay = extract_retry_delay_429(error_msg)

            print(f"[429] Sleep {delay}s")

            time.sleep(delay)

        return None, error_msg


def worker_generate(queue, api_key, contents):

    result = generate_with_client(
        api_key,
        contents
    )

    queue.put(result)


def generate_with_timeout(
    api_key,
    contents,
    timeout=240
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

    if process.is_alive():

        process.terminate()

        process.join()

        return (
            None,
            f"TIMEOUT_AFTER_{timeout}_SECONDS"
        )

    if not queue.empty():
        return queue.get()

    return (
        None,
        "EMPTY_PROCESS_RESULT"
    )


# =========================================================
# Load Existing Results
# =========================================================
save_df = read_jsonl_keep_vid_as_str(
    SAVE_PATH
)

processed_vids = (
    set(save_df["vid"].astype(str))
    if not save_df.empty
    else set()
)


# =========================================================
# Dataset
# =========================================================
class MyDataset(Dataset):

    def __init__(self):

        df = pd.read_json(
            INPUT_PATH,
            lines=True,
            dtype={"vid": str}
        )

        text_refine_df = pd.read_json(
            TEXT_REFINE_PATH,
            lines=True,
            dtype={"vid": str}
        )

        vision_refine_df = pd.read_json(
            VISION_REFINE_PATH,
            lines=True,
            dtype={"vid": str}
        )

        # rename
        text_refine_df = text_refine_df.rename(
            columns={
                "ret": "lm_text_refine"
            }
        )

        vision_refine_df = vision_refine_df.rename(
            columns={
                "ret": "lm_vision_refine"
            }
        )

        # merge
        df = df.merge(
            text_refine_df[
                ["vid", "lm_text_refine"]
            ],
            on="vid",
            how="inner"
        )

        df = df.merge(
            vision_refine_df[
                ["vid", "lm_vision_refine"]
            ],
            on="vid",
            how="inner"
        )

        # remove processed
        df = df[
            ~df["vid"].isin(processed_vids)
        ]

        df = df.reset_index(drop=True)

        self.data = df

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        vid = str(row["vid"])

        dataset_name = str(
            row.get("dataset", "")
        )

        title = str(
            row.get("title", "")
        )

        description = str(
            row.get("description", "")
        )

        label = int(
            row.get("label", -1)
        )

        lm_text_refine = str(
            row.get("lm_text_refine", "")
        )

        lm_vision_refine = str(
            row.get("lm_vision_refine", "")
        )

        title = (
            title + " " + description
        ).strip()

        title = title[:1024]

        text = f"""
title:
{title}

textual_description:
{lm_text_refine}

visual_description:
{lm_vision_refine}
"""

        prompt = PROMPT_TEMPLATE.format(
            text=text
        )

        if "FakeSV" in dataset_name:
            prompt += "\nPlease answer in Chinese."

        return (
            vid,
            dataset_name,
            prompt,
            label
        )


def customed_collate_fn(batch):

    vids, datasets, prompts, labels = zip(*batch)

    return (
        vids,
        datasets,
        prompts,
        labels
    )


# =========================================================
# DataLoader
# =========================================================
dataloader = DataLoader(
    MyDataset(),
    batch_size=1,
    collate_fn=customed_collate_fn,
    num_workers=2,
    shuffle=False
)


# =========================================================
# Thread Control
# =========================================================
lock = threading.Lock()

in_progress_vids = set()


# =========================================================
# Process
# =========================================================
def process_single(args):

    start_time = time.time()

    (
        vid,
        dataset_name,
        prompt,
        label,
        api_key
    ) = args

    with lock:

        if (
            vid in processed_vids
            or vid in in_progress_vids
        ):
            return None

        in_progress_vids.add(vid)

    try:

        contents = [prompt]

        input_tokens = count_input_tokens(
            api_key,
            contents
        )

        result, raw_text = generate_with_timeout(
            api_key=api_key,
            contents=contents,
            timeout=REQUEST_TIMEOUT
        )

        if result is None:

            with lock:

                append_jsonl(
                    FAILED_PATH,
                    {
                        "vid": vid,
                        "dataset": dataset_name,
                        "label": label,
                        "token": input_tokens,
                        "reason": raw_text
                    }
                )

            return None

        elapsed_time = round(
            time.time() - start_time,
            4
        )

        save_obj = {
            "vid": vid,
            "dataset": dataset_name,
            "label": label,
            "token": input_tokens,
            "time": elapsed_time,
            "ret": result["refined_text"]
        }

        debug_obj = {
            "vid": vid,
            "dataset": dataset_name,
            "token": input_tokens,
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
                    "dataset": dataset_name,
                    "label": label,
                    "reason": str(e)
                }
            )

        return None

    finally:

        with lock:
            in_progress_vids.discard(vid)


# =========================================================
# Create Tasks
# =========================================================
tasks = []

idx = 0

for batch in dataloader:

    (
        vids,
        datasets,
        prompts,
        labels
    ) = batch

    for (
        vid,
        dataset_name,
        prompt,
        label
    ) in zip(
        vids,
        datasets,
        prompts,
        labels
    ):

        api_key = API_KEYS[
            idx % len(API_KEYS)
        ]

        tasks.append(
            (
                str(vid),
                dataset_name,
                prompt,
                int(label),
                api_key
            )
        )

        idx += 1


# =========================================================
# Run
# =========================================================
success = 0

with ThreadPoolExecutor(
    max_workers=len(API_KEYS)
) as executor:

    futures = [
        executor.submit(
            process_single,
            task
        )
        for task in tasks
    ]

    for future in tqdm(
        as_completed(futures),
        total=len(futures)
    ):

        try:

            result = future.result()

            if result is not None:
                success += 1

        except Exception as e:

            print(f"Future Error: {str(e)}")


print(f"Success: {success}")