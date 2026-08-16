import os
import sys
import subprocess
import glob
import shutil
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================
# Config
# =========================
NUM_FRAMES = 16

# 建議:
# I/O + ffmpeg 任務通常可以開比 CPU 更多 threads
MAX_WORKERS = min(8, multiprocessing.cpu_count() * 2)

# =========================
# Utility Functions
# =========================
def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        return float(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get video duration {video_path}: {e}")
        return None

    except ValueError:
        logging.error(f"Failed to parse video duration {video_path}")
        return None


def get_video_framerate(video_path):
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'csv=p=0',
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        framerate = result.stdout.strip().split('/')

        return float(framerate[0]) / float(framerate[1])

    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        logging.warning(
            f"Failed to get video framerate {video_path}, "
            f"using default value 30: {e}"
        )
        return 30


# =========================
# Core Processing
# =========================
def extract_frames(video_path, output_folder, num_frames):
    duration = get_video_duration(video_path)

    if duration is None or duration <= 0:
        logging.error(f"Invalid video file or duration: {video_path}")
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    # Skip already processed
    if os.path.exists(video_output_folder):
        existing_frames = glob.glob(
            os.path.join(video_output_folder, "frame_*.jpg")
        )

        if len(existing_frames) == num_frames:
            logging.info(
                f"Skipping {video_name}: "
                f"already has {num_frames} frames"
            )
            return True

        shutil.rmtree(video_output_folder)

    os.makedirs(video_output_folder, exist_ok=True)

    interval = duration / num_frames
    timestamps = [i * interval for i in range(num_frames)]

    # 可選: 讀 fps
    _ = get_video_framerate(video_path)

    success_count = 0

    for i, ts in enumerate(timestamps):
        output_file = os.path.join(
            video_output_folder,
            f"frame_{i:03d}.jpg"
        )

        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-ss', str(ts),
                    '-i', video_path,
                    '-frames:v', '1',
                    '-q:v', '2',
                    output_file
                ],
                check=True,
                capture_output=True,
                text=True
            )

            success_count += 1

        except subprocess.CalledProcessError as e:
            logging.error(
                f"Error processing frame {i} "
                f"of video {video_name}"
            )
            logging.error(f"ffmpeg error: {e.stderr}")

    created_frames = glob.glob(
        os.path.join(video_output_folder, "frame_*.jpg")
    )

    if len(created_frames) != num_frames:
        logging.warning(
            f"{video_name}: only created "
            f"{len(created_frames)}/{num_frames} frames"
        )
        return False

    return True


# =========================
# Multi-thread Processing
# =========================
def process_dataset(dataset_name, num_frames, max_workers):
    input_folder = f'data/{dataset_name}/videos'
    output_folder = f'data/{dataset_name}/frames_{num_frames}_original'

    if not os.path.exists(input_folder):
        logging.error(f"Input folder does not exist: {input_folder}")
        return False

    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, '*.mp4'))

    if not video_paths:
        logging.warning(f"No videos found in {input_folder}")
        return False

    logging.info(
        f"Processing {len(video_paths)} videos "
        f"with {max_workers} threads"
    )

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_video = {
            executor.submit(
                extract_frames,
                video_path,
                output_folder,
                num_frames
            ): video_path
            for video_path in video_paths
        }

        for future in tqdm(
            as_completed(future_to_video),
            total=len(video_paths),
            desc=f"Processing {dataset_name}"
        ):

            video_path = future_to_video[future]

            try:
                result = future.result()

                if result:
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                logging.error(
                    f"Unhandled exception for {video_path}: {e}"
                )

    logging.info(
        f"{dataset_name} finished | "
        f"Success: {success}, Failed: {failed}"
    )

    return success > 0


# =========================
# Main
# =========================
def main():

    datasets = ["ALL"]

    processed_count = 0

    for dataset in datasets:

        ok = process_dataset(
            dataset_name=dataset,
            num_frames=NUM_FRAMES,
            max_workers=MAX_WORKERS
        )

        if ok:
            processed_count += 1

    if processed_count == 0:
        logging.error("No datasets were processed successfully")
        sys.exit(1)

    logging.info(
        f"Successfully processed "
        f"{processed_count}/{len(datasets)} datasets"
    )


if __name__ == "__main__":
    main()