import os
import glob
import shutil
import logging
import concurrent.futures
import functools

import cv2
import numpy as np

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =========================
# Config
# =========================

# 原始 frame
FRAME_ROOT = "data/ALL/frames"

# 最終 16 張 frame
FRAME16_ROOT = "data/ALL/frames_16"

MAX_FRAMES = 16

SSIM_THRESHOLD = 0.99

MAX_WORKERS = 6


# =========================
# Utils
# =========================

def load_image_gray(image_path):
    """
    讀取灰階圖片
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray


def compute_ssim(img1_path, img2_path):
    """
    計算兩張圖片 SSIM
    """

    img1 = load_image_gray(img1_path)

    img2 = load_image_gray(img2_path)

    if img1 is None or img2 is None:
        return 0

    # resize 避免尺寸不同
    if img1.shape != img2.shape:

        img2 = cv2.resize(
            img2,
            (img1.shape[1], img1.shape[0])
        )

    score = ssim(img1, img2)

    return score


# =========================
# Rename
# =========================

def rename_frames_inplace(frame_paths, folder_path):
    """
    重新命名：

    frame_1.jpg
    frame_2.jpg
    ...
    """

    temp_paths = []

    # ---------- temp ----------
    for idx, frame_path in enumerate(frame_paths):

        temp_path = os.path.join(
            folder_path,
            f"temp_{idx + 1}.jpg"
        )

        shutil.copy(
            frame_path,
            temp_path
        )

        temp_paths.append(temp_path)

    # ---------- remove old ----------
    old_frames = glob.glob(
        os.path.join(folder_path, "frame_*.jpg")
    )

    for old_file in old_frames:

        if os.path.exists(old_file):
            os.remove(old_file)

    # ---------- temp -> frame ----------
    for idx, temp_path in enumerate(temp_paths):

        new_path = os.path.join(
            folder_path,
            f"frame_{idx + 1}.jpg"
        )

        os.rename(
            temp_path,
            new_path
        )


# =========================
# SSIM Filtering
# =========================

def remove_high_ssim_frames(frame_paths):
    """
    移除重複性過高 frame

    ```
    保留：
    - 第一張
    - 與上一張差異夠大的 frame
    - 最後一張

    所有影片都做 SSIM filtering
    """

    # frame 太少
    if len(frame_paths) <= 2:
        return frame_paths

    selected = [frame_paths[0]]

    prev_frame = frame_paths[0]

    # 中間 frame
    for current_frame in frame_paths[1:-1]:

        score = compute_ssim(
            prev_frame,
            current_frame
        )

        # 差異夠大才保留
        if score < SSIM_THRESHOLD:

            selected.append(current_frame)

            prev_frame = current_frame

    # 最後一張一定保留
    if frame_paths[-1] != selected[-1]:

        selected.append(frame_paths[-1])

    return selected

# =========================
# Uniform Sampling
# =========================

def uniform_sample_frames(frame_paths):
    """
    等間距取樣固定 16 張

    保留：
    - 第一張
    - 最後一張
    """

    if len(frame_paths) <= MAX_FRAMES:
        return frame_paths

    indices = np.linspace(
        0,
        len(frame_paths) - 1,
        MAX_FRAMES,
        dtype=int
    )

    sampled = [
        frame_paths[idx]
        for idx in indices
    ]

    return sampled


# =========================
# Save 16 Frames
# =========================

def save_frames_16(
    sampled_frames,
    output_folder
):
    """
    將最終 16 張 frame
    存到 frames_16
    """

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # 清空舊資料
    old_frames = glob.glob(
        os.path.join(output_folder, "frame_*.jpg")
    )

    for old_file in old_frames:

        if os.path.exists(old_file):
            os.remove(old_file)

    # 存新 frame
    for idx, frame_path in enumerate(sampled_frames):

        output_path = os.path.join(
            output_folder,
            f"frame_{idx + 1}.jpg"
        )

        shutil.copy(
            frame_path,
            output_path
        )


# =========================
# Main Per Video
# =========================

def process_video_folder(folder_path):

    video_name = os.path.basename(folder_path)

    frame_paths = sorted(
        glob.glob(
            os.path.join(folder_path, "frame_*.jpg")
        )
    )

    if len(frame_paths) == 0:

        logging.warning(
            f"No frames: {folder_path}"
        )

        return

    original_count = len(frame_paths)

    # ====================================
    # Step 1
    # SSIM filtering
    # ====================================

    filtered_frames = remove_high_ssim_frames(
        frame_paths
    )

    # ====================================
    # Step 2
    # 原本 frames 只保留 SSIM 後結果
    # ====================================

    rename_frames_inplace(
        filtered_frames,
        folder_path
    )

    # 重新取得最新 frame
    filtered_frame_paths = sorted(
        glob.glob(
            os.path.join(folder_path, "frame_*.jpg")
        )
    )

    # ====================================
    # Step 3
    # Uniform sampling -> 16 frames
    # ====================================

    sampled_frames = uniform_sample_frames(
        filtered_frame_paths
    )

    # ====================================
    # Step 4
    # save to frames_16
    # ====================================

    output_16_folder = os.path.join(
        FRAME16_ROOT,
        video_name
    )

    save_frames_16(
        sampled_frames,
        output_16_folder
    )

    logging.info(
        f"{video_name} | "
        f"{original_count} -> "
        f"{len(filtered_frames)} -> "
        f"{len(sampled_frames)}"
    )


# =========================
# Main
# =========================

def main():

    if not os.path.exists(FRAME_ROOT):

        logging.error(
            f"Frame root not found: "
            f"{FRAME_ROOT}"
        )

        return

    os.makedirs(
        FRAME16_ROOT,
        exist_ok=True
    )

    video_folders = sorted([

        os.path.join(FRAME_ROOT, d)

        for d in os.listdir(FRAME_ROOT)

        if os.path.isdir(
            os.path.join(FRAME_ROOT, d)
        )
    ])

    logging.info(
        f"Total folders: "
        f"{len(video_folders)}"
    )

    func = functools.partial(
        process_video_folder
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        list(
            tqdm(
                executor.map(
                    func,
                    video_folders
                ),
                total=len(video_folders),
                desc="Processing Frames"
            )
        )

    logging.info("Done")


# =========================
# Entry
# =========================

if __name__ == "__main__":

    main()