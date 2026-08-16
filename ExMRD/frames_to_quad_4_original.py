import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import traceback

# =========================
# Config
# =========================
NUM_FRAMES = 16

# I/O 任務適合多 threads
MAX_WORKERS = min(16, multiprocessing.cpu_count() * 2)

# =========================
# Single Folder Processing
# =========================
def process_single_video_folder(
    vid_folder,
    input_folder,
    output_folder
):
    try:
        frames_folder = os.path.join(input_folder, vid_folder)

        frames = []

        # =========================
        # Read Frames
        # =========================
        for i in range(NUM_FRAMES):

            frame_path = os.path.join(
                frames_folder,
                f"frame_{i:03d}.jpg"
            )

            if os.path.exists(frame_path):

                try:
                    frame = Image.open(frame_path).convert("RGB")
                    frames.append(frame)

                except Exception as e:
                    return False, (
                        f"Failed to open image {frame_path}: {e}"
                    )

            else:
                return False, f"Cannot find frame: {frame_path}"

        if len(frames) < NUM_FRAMES:
            return False, (
                f"Folder {vid_folder} contains fewer "
                f"than {NUM_FRAMES} frames"
            )

        # =========================
        # Create 4 quad images
        # =========================
        for quad_index in range(4):

            start_index = quad_index * 4
            quad_frames = frames[start_index:start_index + 4]

            min_width = min(frame.width for frame in quad_frames)
            min_height = min(frame.height for frame in quad_frames)

            quad_frames = [
                frame.resize((min_width, min_height))
                for frame in quad_frames
            ]

            grid = Image.new(
                'RGB',
                (min_width * 2, min_height * 2)
            )

            # 2x2 layout
            grid.paste(quad_frames[0], (0, 0))
            grid.paste(quad_frames[1], (min_width, 0))
            grid.paste(quad_frames[2], (0, min_height))
            grid.paste(quad_frames[3], (min_width, min_height))

            output_path = os.path.join(
                output_folder,
                f"{vid_folder}_quad_{quad_index}.jpg"
            )

            grid.save(
                output_path,
                'JPEG',
                quality=95
            )

        return True, vid_folder

    except Exception as e:

        error_msg = (
            f"Unhandled exception in {vid_folder}\n"
            f"{str(e)}\n"
            f"{traceback.format_exc()}"
        )

        return False, error_msg


# =========================
# Dataset Processing
# =========================
def process_frames(
    input_folder,
    output_folder,
    max_workers=MAX_WORKERS
):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_folders = [
        f for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f))
    ]

    print(f"Found {len(video_folders)} video folders")
    print(f"Using {max_workers} threads")

    processed_count = 0
    error_count = 0

    # =========================
    # Multi-thread Processing
    # =========================
    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                process_single_video_folder,
                vid_folder,
                input_folder,
                output_folder
            ): vid_folder
            for vid_folder in video_folders
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing video folders",
            unit="folder"
        ):

            vid_folder = futures[future]

            try:
                success, message = future.result()

                if success:
                    processed_count += 1

                else:
                    error_count += 1
                    tqdm.write(f"[ERROR] {message}")

            except Exception as e:

                error_count += 1

                tqdm.write(
                    f"[CRASH] {vid_folder}: {str(e)}"
                )

    print("\n=========================")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {error_count}")
    print("=========================")

    return processed_count > 0


# =========================
# Main
# =========================
def main():

    datasets = ["ALL"]

    total_success = 0

    for dataset in datasets:

        print(f"\nProcessing dataset: {dataset}")

        input_folder = (
            f'data/{dataset}/frames_16_original'
        )

        output_folder = (
            f'data/{dataset}/quads_4_original'
        )

        if not os.path.exists(input_folder):

            print(
                f"[ERROR] Input folder does not exist:\n"
                f"{input_folder}"
            )

            continue

        success = process_frames(
            input_folder=input_folder,
            output_folder=output_folder,
            max_workers=MAX_WORKERS
        )

        if success:
            total_success += 1

    print("\n=========================")
    print(
        f"Finished processing "
        f"{total_success}/{len(datasets)} datasets"
    )
    print("=========================")


if __name__ == "__main__":
    main()