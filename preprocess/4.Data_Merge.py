import os
import json
import pandas as pd


# =========================
# Config
# =========================

ROOT_DIR = "data/ALL"

DATA_FILE = os.path.join(ROOT_DIR, "data.jsonl")
OCR_FILE = os.path.join(ROOT_DIR, "ocr.jsonl")
TRANSCRIPT_FILE = os.path.join(ROOT_DIR, "transcript.jsonl")

OUTPUT_FILE = os.path.join(ROOT_DIR, "all.jsonl")


# =========================
# Utils
# =========================

def load_jsonl(path):
    """
    讀取 jsonl
    並將 vid 強制轉成 string
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_json(
        path,
        lines=True,
        dtype={"vid": str}
    )

    # 保險處理
    df["vid"] = df["vid"].astype(str)

    return df


# =========================
# Main
# =========================

def main():

    print("Loading files...")

    # ---------- load ----------
    data_df = load_jsonl(DATA_FILE)
    ocr_df = load_jsonl(OCR_FILE)
    transcript_df = load_jsonl(TRANSCRIPT_FILE)

    print(f"data.jsonl: {len(data_df)}")
    print(f"ocr.jsonl: {len(ocr_df)}")
    print(f"transcript.jsonl: {len(transcript_df)}")

    # ---------- merge OCR ----------
    merged_df = data_df.merge(
        ocr_df,
        on="vid",
        how="left"
    )

    # ---------- merge transcript ----------
    merged_df = merged_df.merge(
        transcript_df,
        on="vid",
        how="left"
    )

    # ---------- fill missing ----------
    if "ocr" not in merged_df.columns:
        merged_df["ocr"] = ""

    if "transcript" not in merged_df.columns:
        merged_df["transcript"] = ""

    merged_df["ocr"] = merged_df["ocr"].fillna("")
    merged_df["transcript"] = merged_df["transcript"].fillna("")

    # ---------- keep columns ----------
    keep_columns = [
        "vid",
        "label",
        "dataset",
        "title",
        "ocr",
        "transcript"
    ]

    # 若缺欄位則補空字串
    for col in keep_columns:

        if col not in merged_df.columns:
            merged_df[col] = ""

    merged_df = merged_df[keep_columns]

    # ---------- save ----------
    print("Saving all.jsonl...")

    merged_df.to_json(
        OUTPUT_FILE,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total records: {len(merged_df)}")


# =========================
# Entry
# =========================

if __name__ == "__main__":

    main()