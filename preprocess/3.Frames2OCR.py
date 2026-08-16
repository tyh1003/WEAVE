import os
import glob
import json
import re

import cv2
import pandas as pd

from tqdm import tqdm
from Levenshtein import ratio
from paddleocr import PaddleOCR


# =========================
# Config
# =========================

DATASET = "ALL"

FRAME_ROOT = f"data/{DATASET}/frames"

OUTPUT_FILE = f"data/{DATASET}/ocr.jsonl"

LABEL_FILE = f"data/{DATASET}/label.jsonl"


# =========================
# PaddleOCR
# =========================

ocr_model = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    show_log=False
)


# =========================
# Utils
# =========================

def load_image(image_path):
    """
    讀取圖片
    """

    image = cv2.imread(image_path)

    return image


def clean_text(text):
    """
    清理 OCR text

    保留：
    - 中文
    - 英文
    - 數字
    - 常見符號
    """

    text = re.sub(
        r'[^\u4e00-\u9fffA-Za-z0-9\s\.\,\:\%\$\#\@\-\_\/]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# =========================
# OCR
# =========================

def ocr_single_image(
    image_path,
    confidence_threshold=0.5
):
    """
    OCR 單張圖片
    """

    image = load_image(image_path)

    if image is None:
        return ""

    try:

        result = ocr_model.ocr(
            image,
            cls=True
        )

        if result is None:
            return ""

        texts = []

        for line in result:

            if line is None:
                continue

            for item in line:

                if item is None:
                    continue

                text = item[1][0]

                confidence = item[1][1]

                # confidence filtering
                if confidence < confidence_threshold:
                    continue

                text = clean_text(text)

                if len(text) > 0:
                    texts.append(text)

        merged_text = ' '.join(texts).strip()

        return merged_text

    except Exception:

        return ""


# =========================
# Remove Duplicate OCR
# =========================

def remove_duplicate_texts(
    texts,
    threshold=0.7
):
    """
    去除重複 OCR text
    """

    if not texts:
        return []

    unique = [texts[0]]

    for i in range(1, len(texts)):

        prev_text = unique[-1]

        current_text = texts[i]

        similarity = ratio(
            prev_text,
            current_text
        )

        if similarity < threshold:

            unique.append(current_text)

    return unique


# =========================
# OCR One Video
# =========================

def extract_text_from_frames(frame_folder):
    """
    對單支影片 frame 做 OCR
    """

    frame_paths = sorted(
        glob.glob(
            os.path.join(
                frame_folder,
                "frame_*.jpg"
            )
        )
    )

    if len(frame_paths) == 0:
        return []

    texts = []

    for frame_path in frame_paths:

        text = ocr_single_image(frame_path)

        if len(text) > 3:

            texts.append(text)

    # remove duplicated OCR
    texts = remove_duplicate_texts(texts)

    return texts


# =========================
# Main
# =========================

def main():

    if not os.path.exists(LABEL_FILE):

        print(
            f"Label file not found: "
            f"{LABEL_FILE}"
        )

        return

    if not os.path.exists(FRAME_ROOT):

        print(
            f"Frame root not found: "
            f"{FRAME_ROOT}"
        )

        return

    # ---------- 讀 label ----------
    df = pd.read_json(
        LABEL_FILE,
        lines=True,
        dtype={'vid': str}
    )

    vid_list = df['vid'].tolist()

    # ---------- 已完成 ----------
    processed_ids = set()

    if os.path.exists(OUTPUT_FILE):

        with open(
            OUTPUT_FILE,
            'r',
            encoding='utf-8'
        ) as f:

            for line in f:

                try:

                    data = json.loads(line)

                    processed_ids.add(
                        data['vid']
                    )

                except:
                    continue

    # ---------- 開始 OCR ----------
    with open(
        OUTPUT_FILE,
        'a',
        encoding='utf-8'
    ) as out_file:

        for vid in tqdm(
            vid_list,
            desc="Processing OCR"
        ):

            # skip processed
            if vid in processed_ids:
                continue

            frame_folder = os.path.join(
                FRAME_ROOT,
                vid
            )

            if not os.path.exists(frame_folder):
                continue

            texts = extract_text_from_frames(
                frame_folder
            )

            ocr_text = '\n'.join(texts)

            record = {
                'vid': vid,
                'ocr': ocr_text
            }

            out_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + '\n'
            )

            out_file.flush()

            processed_ids.add(vid)

    print("OCR complete")


# =========================
# Entry
# =========================

if __name__ == "__main__":

    main()