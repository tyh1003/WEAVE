import json
import os
import random
import re
import time
import dspy
import litellm
import pandas as pd
from dotenv import load_dotenv
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate 
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. Configurations
# ==========================================

INPUT_PATH = "data/All/NEW/retrieve.jsonl"
ALL_PATH = "data/All/all.jsonl"
OUTPUT_DIR = "data/All/DSPy/"

OPTIMIZED_PROGRAM_PATH = os.path.join(OUTPUT_DIR, "optimized_rag_judge.json")
PRED_PATH = os.path.join(OUTPUT_DIR, "DSPy.jsonl")
DEBUG_PATH = os.path.join(OUTPUT_DIR, "DSPy_debug.jsonl")
RECORD_PATH = os.path.join(OUTPUT_DIR, "record.jsonl")
PROMPT_HISTORY_PATH = os.path.join(OUTPUT_DIR, "prompt_history.jsonl")
BEST_PROMPT_PATH = os.path.join(OUTPUT_DIR, "best_prompt.txt")
TEST_EVAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "test_evaluation_report.json") 

MODEL_NAME = "gemini/gemma-4-31b-it"
SEED = 42

TRAIN_SIZE = 500
VAL_SIZE = 200
TEST_SIZE = 200
TOTAL_REQUIRED = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

MAX_EVAL_SIZE = None
MAX_RETRY = 5
NUM_THREADS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
load_dotenv()
random.seed(SEED)

# ==========================================
# 2. LM Config (新版相容：標準繼承與重試機制)
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY_10")
if not API_KEY:
    raise RuntimeError("請在 .env 中設定有效的 GEMINI_API_KEY_10")


class RetryLM(dspy.LM):
    """
    正統繼承自 dspy.LM 的自訂類別，完美通過 isinstance 檢查。
    相容新版 DSPy 的基底類別設計，透過 super().__call__ 攔截 RateLimitError。
    """
    def __init__(self, model_name, **kwargs):
        os.environ["GEMINI_API_KEY"] = API_KEY
        super().__init__(model=model_name, **kwargs)

    def __call__(self, *args, **kwargs):
        while True:
            try:
                return super().__call__(*args, **kwargs)
            
            except litellm.RateLimitError as e:
                error_msg = str(e)
                print("\n[RateLimit] Gemini quota exceeded, preparing to sleep...")
                
                sleep_seconds = 60.0

                try:
                    json_match = re.search(r"geminiException -\s*(\{.*\})", error_msg, re.DOTALL)
                    if json_match:
                        error_data = json.loads(json_match.group(1))
                        details = error_data.get("error", {}).get("details", [])
                        for detail in details:
                            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                retry_delay = detail.get("retryDelay", "")
                                sleep_seconds = float(retry_delay.replace("s", ""))
                                print(f"[RetryDelay JSON] 成功提取等待時間: {sleep_seconds} 秒")
                                break
                    else:
                        regex_match = re.search(r"Please retry in ([\d\.]+)s", error_msg)
                        if regex_match:
                            sleep_seconds = float(regex_match.group(1))
                            print(f"[RetryDelay Regex] 成功提取等待時間: {sleep_seconds} 秒")
                except Exception as parse_error:
                    print(f"[RetryDelay Parse Failed] 解析失敗 ({parse_error})，將使用預設時間。")

                final_sleep = sleep_seconds + 2.0
                print(f"[Sleep] 程式將暫停 {final_sleep:.2f} 秒後自動重試...\n")
                time.sleep(final_sleep)


def configure_new_lm():
    lm = RetryLM(model_name=MODEL_NAME, temperature=0.0, max_tokens=4096)
    dspy.configure(lm=lm)
    return lm

current_lm = configure_new_lm()

# ==========================================
# 3. Utils
# ==========================================

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


def safe_prob(x):
    try:
        x = round(float(x), 4)
        return max(0.0, min(1.0, x))
    except:
        return 0.5


def prob_to_label(prob, threshold=0.5):
    return 1 if prob > threshold else 0


def build_background_knowledge(k_int, k_ext):
    k_int = str(k_int).strip()
    k_ext = str(k_ext).strip()

    knowledge_parts = []
    if k_int:
        knowledge_parts.append(k_int)
    if k_ext:
        knowledge_parts.append(k_ext)

    if len(knowledge_parts) == 0:
        return "無"

    return "\n".join(knowledge_parts)


def row_to_example(row):
    return dspy.Example(
        Rc=str(row["Rc"]).strip(),
        Rv=str(row["Rv"]).strip(),
        background_knowledge=str(row["background_knowledge"]).strip(),
        fake_prob=float(row["label"]),
    ).with_inputs("Rc", "Rv", "background_knowledge")


# ==========================================
# 4. Dataset
# ==========================================

df = read_jsonl_keep_vid_as_str(INPUT_PATH)
if df.empty:
    raise RuntimeError(f"{INPUT_PATH} empty")

required_cols = ["vid", "label", "Rc", "Rv", "K_int", "K_ext"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"Missing columns: {missing_cols}")

df["vid"] = df["vid"].astype(str)
df["label"] = df["label"].astype(int)
df["background_knowledge"] = df.apply(
    lambda row: build_background_knowledge(row["K_int"], row["K_ext"]), axis=1
)
df = df.reset_index(drop=True)

all_df = read_jsonl_keep_vid_as_str(ALL_PATH)
if not all_df.empty and "dataset" in all_df.columns:
    all_df["vid"] = all_df["vid"].astype(str)
    df = df.merge(all_df[["vid", "dataset"]], on="vid", how="left")
else:
    df["dataset"] = "UNKNOWN"

stratify_col = df["label"] if df["label"].nunique() == 2 else None

if len(df) > TOTAL_REQUIRED:
    target_df, _ = train_test_split(
        df, train_size=TOTAL_REQUIRED, random_state=SEED, stratify=stratify_col
    )
else:
    target_df = df

train_df, temp_df = train_test_split(
    target_df, train_size=TRAIN_SIZE, random_state=SEED, stratify=target_df["label"]
)

val_df, test_df = train_test_split(
    temp_df, train_size=VAL_SIZE, random_state=SEED, stratify=temp_df["label"]
)

# 轉換成 DSPy Example 格式
trainset = [row_to_example(row) for _, row in train_df.iterrows()]
valset = [row_to_example(row) for _, row in val_df.iterrows()]    # 改動：原為 devset，更名為 valset
testset = [row_to_example(row) for _, row in test_df.iterrows()]  # 新增：建立測試集物件

print(f"Train size: {len(trainset)}")
print(f"Val size: {len(valset)}")
print(f"Test size: {len(testset)}")


# ==========================================
# 5. DSPy Signature
# ==========================================

class RAGFakeNewsJudge(dspy.Signature):
    """
    你是一個嚴格的RAG假訊息判斷系統，需要嚴格根據背景知識判斷短影音真假。

    限制：
    - pred_label 為 0~1 之間的機率值
    - 保留四位小數
    - 越接近1越假
    - 越接近0越真
    - 0.5代表無法判斷
    - reason 必須使用繁體中文
    """

    Rc = dspy.InputField(desc="文字輸入核心內容")
    Rv = dspy.InputField(desc="影像輸入核心內容")
    background_knowledge = dspy.InputField(desc="背景知識")
    pred_label = dspy.OutputField(desc="短影音為假的機率")
    reason = dspy.OutputField(desc="判斷原因")


class RAGJudgeModule(dspy.Module):

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(RAGFakeNewsJudge)

    def forward(self, Rc, Rv, background_knowledge):
        res = self.generate_answer(
            Rc=Rc, Rv=Rv, background_knowledge=background_knowledge
        )
        
        # 存答案
        return dspy.Prediction(
            pred_label=safe_prob(getattr(res, "pred_label", 0.5)),
            reason=str(getattr(res, "reason", "")).strip(),
        )


# ==========================================
# 6. Metric
# ==========================================

def judge_metric(example, pred, trace=None):
    gold = float(example.fake_prob)
    pred_prob = safe_prob(getattr(pred, "pred_label", 0.5))
    reason = str(getattr(pred, "reason", "")).strip()

    mse_score = 0.95 * (1.0 - ((gold - pred_prob) ** 2))
    reason_score = 0.05 if len(reason) >= 50 else 0.0
    score = mse_score + reason_score

    return max(0.0, min(score, 1.0))


# ==========================================
# 7. MIPROv2
# ==========================================

student = RAGJudgeModule()

optimizer = MIPROv2(
    metric=judge_metric,
    prompt_model=current_lm,
    task_model=current_lm,
    auto="medium"
)

print("\n===== 開始進行 MIPROv2 最佳化編譯 =====")

if os.path.exists(PROMPT_HISTORY_PATH):
    os.remove(PROMPT_HISTORY_PATH)

if hasattr(current_lm, "history"):
    current_lm.history.clear()

# 修改：在 compile 內同時傳入 trainset 與 valset
optimized_program = optimizer.compile(
    student,
    trainset=trainset,
    valset=valset
)

optimized_program.save(OPTIMIZED_PROGRAM_PATH)

print(f"\nOptimized program saved:\n{OPTIMIZED_PROGRAM_PATH}")


# ==========================================
# 8. Save Prompt History
# ==========================================

print("\n===== Saving Prompt History =====")
all_prompts = []

if hasattr(current_lm, "history"):
    for idx, item in enumerate(current_lm.history):
        prompt_text = None

        if isinstance(item, dict):
            if "prompt" in item and item["prompt"]:
                prompt_text = item["prompt"]
            elif "messages" in item and item["messages"]:
                prompt_text = json.dumps(item["messages"], ensure_ascii=False, indent=2)
        elif hasattr(item, "prompt"):
            prompt_text = item.prompt
        elif hasattr(item, "request"):
            req = item.request
            if isinstance(req, dict):
                if "prompt" in req:
                    prompt_text = req["prompt"]
                elif "messages" in req:
                    prompt_text = json.dumps(
                        req["messages"], ensure_ascii=False, indent=2
                    )

        if prompt_text:
            all_prompts.append(prompt_text)
            append_jsonl(PROMPT_HISTORY_PATH, {"idx": idx, "prompt": prompt_text})

print(f"Saved {len(all_prompts)} prompts")


# ==========================================
# 9. Save Best Prompt (完整版)
# ==========================================

RAW_OPTIMIZED_JSON_PATH  = os.path.join(OUTPUT_DIR, "optimized_program_full.json")
FULL_RUNTIME_PROMPT_PATH = os.path.join(OUTPUT_DIR, "full_runtime_prompt.txt")


def recursive_find(obj, keywords, path=""):
    """遞迴搜尋 JSON 結構中包含指定關鍵字的鍵值對"""
    results = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            if any(kw.lower() in k.lower() for kw in keywords):
                results.append((current_path, v))
            results.extend(recursive_find(v, keywords, current_path))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            current_path = f"{path}[{i}]"
            results.extend(recursive_find(v, keywords, current_path))

    return results


print("\n===== Saving Best Prompt =====")

# -------------------------------------------------------------------------
# 1. 載入並儲存原始的 Optimized JSON
# -------------------------------------------------------------------------
with open(OPTIMIZED_PROGRAM_PATH, "r", encoding="utf-8") as f:
    optimized_json = json.load(f)

with open(RAW_OPTIMIZED_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(optimized_json, f, ensure_ascii=False, indent=2)

print(f"Full optimized json saved:\n{RAW_OPTIMIZED_JSON_PATH}")

# -------------------------------------------------------------------------
# 2. 搜尋所有可能的重要 Prompt 元件並儲存
# -------------------------------------------------------------------------
keywords = ("instruction", "prompt", "prefix", "desc", "demo", "example", "trace", "signature")
items = recursive_find(optimized_json, keywords)

with open(BEST_PROMPT_PATH, "w", encoding="utf-8") as f:
    f.write("====================================================\n")
    f.write("DSPY OPTIMIZED PROMPT COMPONENTS\n")
    f.write("====================================================\n\n")

    for path, value in items:
        f.write(f"\n===== {path} =====\n")
        if isinstance(value, (dict, list)):
            f.write(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            f.write(str(value))
        f.write("\n")

print(f"Prompt components saved:\n{BEST_PROMPT_PATH}")

# -------------------------------------------------------------------------
# 3. 嘗試抽取真正的 Instruction 與欄位資訊
# -------------------------------------------------------------------------
instruction = ""
instruction_candidates = recursive_find(optimized_json, ("instruction",))

if instruction_candidates:
    instruction = str(instruction_candidates[0][1]).strip()

prefix_map = {}
description_map = {}
signature_fields = recursive_find(optimized_json, ("fields",))

# 尋找真正的 fields 結構
fields_obj = None
for path, value in signature_fields:
    if isinstance(value, list) and len(value) > 0:
        fields_obj = value
        break

if fields_obj:
    for field in fields_obj:
        prefix = field.get("prefix", "").strip()
        desc = field.get("description", "").strip()
        if prefix:
            prefix_map[prefix] = desc

# -------------------------------------------------------------------------
# 4. 組合「真正可直接使用」的 Runtime Prompt
# -------------------------------------------------------------------------
runtime_prompt = f"""{instruction}

========================
輸入資料
========================
Rc:
{{Rc}}

Rv:
{{Rv}}

Background Knowledge:
{{background_knowledge}}

========================
輸出格式
========================
Pred Label:
請輸出 0.0000 ~ 1.0000 之間的數值

Reason:
請使用繁體中文簡潔說明理由""".strip()

# -------------------------------------------------------------------------
# 5. 加入 Few-shot / Demos 範例
# -------------------------------------------------------------------------
demo_candidates = recursive_find(optimized_json, ("demo", "example"))

if demo_candidates:
    runtime_prompt += "\n\n========================\n"
    runtime_prompt += "Few-shot Examples\n"
    runtime_prompt += "========================\n"

    for idx, (path, value) in enumerate(demo_candidates[:5]):
        runtime_prompt += f"\n\n----- Example {idx+1} -----\n"
        if isinstance(value, (dict, list)):
            runtime_prompt += json.dumps(value, ensure_ascii=False, indent=2)
        else:
            runtime_prompt += str(value)

# -------------------------------------------------------------------------
# 6. 儲存與終端機輸出
# -------------------------------------------------------------------------
with open(FULL_RUNTIME_PROMPT_PATH, "w", encoding="utf-8") as f:
    f.write(runtime_prompt)

print(f"Full runtime prompt saved:\n{FULL_RUNTIME_PROMPT_PATH}")

print("\n===== FINAL RUNTIME PROMPT =====\n")
print(runtime_prompt[:4000])  # 避免 terminal 爆掉

# ==========================================
# 10. Test Set Evaluation (新增：測試集評估與紀錄)
# ==========================================

print("\n===== 開始在 Test Set 進行最終評估 =====")

# 使用 DSPy 的 Evaluate 模組來批次跑完測試集
evaluator = Evaluate(
    devset=testset, 
    metric=judge_metric, 
    num_threads=NUM_THREADS, 
    display_progress=True
)

eval_result = evaluator(optimized_program)

print(eval_result)

avg_score = eval_result.score   # 或 average_score

print(f"Test Set 平均自訂指標得分 (Judge Metric): {avg_score:.4f}")

# 收集詳細預測結果，用來計算標準的 Machine Learning 分類與機率指標
y_true = []
y_pred_prob = []
y_pred_label = []

if os.path.exists(PRED_PATH):
    os.remove(PRED_PATH)
if os.path.exists(DEBUG_PATH):
    os.remove(DEBUG_PATH)

print("正在彙整預測結果並計算分類指標...")
for idx, sample in enumerate(tqdm(testset, desc="Evaluating details")):
    # 預測
    pred = optimized_program(
        Rc=sample.Rc, 
        Rv=sample.Rv, 
        background_knowledge=sample.background_knowledge
    )
    
    gold_label = int(sample.fake_prob)  # 真實二元標籤
    pred_prob = safe_prob(getattr(pred, "pred_label", 0.5))
    pred_label = prob_to_label(pred_prob)
    
    y_true.append(gold_label)
    y_pred_prob.append(pred_prob)
    y_pred_label.append(pred_label)
    
    # 寫入預測結果 JSONL (對齊你原來的 PRED_PATH 與 DEBUG_PATH 規劃)
    out_obj = {
        "Rc": sample.Rc,
        "Rv": sample.Rv,
        "background_knowledge": sample.background_knowledge,
        "gold_label": gold_label,
        "pred_prob": pred_prob,
        "pred_label": pred_label,
        "reason": getattr(pred, "reason", "")
    }
    append_jsonl(PRED_PATH, out_obj)
    append_jsonl(DEBUG_PATH, {"idx": idx, "prediction": out_obj})

# 計算標準機器學習評估指標
acc = accuracy_score(y_true, y_pred_label)
f1 = f1_score(y_true, y_pred_label, average="binary", pos_label=1)
brier = brier_score_loss(y_true, y_pred_prob)

# 防止機率值極端導致 log_loss 報錯，進行微小剪裁
clipped_probs = [max(1e-15, min(1 - 1e-15, p)) for p in y_pred_prob]
logloss = log_loss(y_true, clipped_probs)

report = {
    "test_size": len(testset),
    "judge_metric_avg_score": round(avg_score, 4),
    "accuracy": round(acc, 4),
    "f1_score_binary_pos_1": round(f1, 4),
    "brier_score_loss": round(brier, 4),
    "log_loss": round(logloss, 4)
}

# 儲存報告
with open(TEST_EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n===== 測試集評估報告 =====")
for k, v in report.items():
    print(f"{k}: {v}")
print(f"測試集詳細報告已儲存至: {TEST_EVAL_REPORT_PATH}")