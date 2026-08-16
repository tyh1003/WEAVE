import os
import sys
import subprocess
import glob
import shutil
import logging
from tqdm import tqdm
import concurrent.futures
import functools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_video_duration(video_path):
    """
    取得影片長度（秒）
    """
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

    except Exception as e:
        logging.error(f"ffprobe failed: {video_path} | {e}")
        return None


def generate_timestamps(duration):
    """
    從第 0 秒開始，每秒取一張
    並確保最後一張 frame 一定包含影片最後時間點

    範例:
    duration = 5.3
    -> [0,1,2,3,4,5,5.3]
    """

    timestamps = []

    current = 0

    while current < int(duration):
        timestamps.append(float(current))
        current += 1

    # 最後一張一定加入影片尾端
    if duration not in timestamps:
        timestamps.append(duration)

    return timestamps


def extract_frames_every_second(video_path, output_folder):
    """
    每秒擷取一張 frame
    並保留最後一張 frame
    使用原始畫質輸出
    """

    duration = get_video_duration(video_path)

    if duration is None or duration <= 0:
        logging.error(f"Invalid video: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_output_folder = os.path.join(output_folder, video_name)

    # 若已存在則略過
    if os.path.exists(video_output_folder):
        existing = glob.glob(
            os.path.join(video_output_folder, "frame_*.jpg")
        )

        expected_frames = len(generate_timestamps(duration))

        if len(existing) == expected_frames:
            logging.info(f"Skip: {video_name}")
            return

        shutil.rmtree(video_output_folder)

    os.makedirs(video_output_folder, exist_ok=True)

    timestamps = generate_timestamps(duration)

    for idx, ts in enumerate(timestamps):

        output_file = os.path.join(
            video_output_folder,
            f"frame_{idx:04d}.jpg"
        )

        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'error',

                    # seek
                    '-ss', str(ts),

                    '-i', video_path,

                    # 只取一張
                    '-frames:v', '1',

                    # 原始畫質
                    '-q:v', '1',

                    output_file
                ],
                check=True
            )

        except subprocess.CalledProcessError:
            logging.warning(
                f"{video_name}: failed at timestamp {ts}"
            )

    extracted = glob.glob(
        os.path.join(video_output_folder, "frame_*.jpg")
    )

    if len(extracted) == 0:
        logging.error(f"{video_name}: no frames extracted")
    else:
        logging.info(
            f"{video_name}: extracted {len(extracted)} frames"
        )


def process_all_videos(max_workers=6):

    input_folder = 'data/ALL/videos'
    output_folder = 'data/ALL/frames'

    if not os.path.exists(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return False

    os.makedirs(output_folder, exist_ok=True)

    # 支援 mp4 / mov / mkv / avi
    extensions = ['*.mp4', '*.mov', '*.mkv', '*.avi']

    video_paths = []

    for ext in extensions:
        video_paths.extend(
            glob.glob(os.path.join(input_folder, ext))
        )

    video_paths = sorted(video_paths)

    if len(video_paths) == 0:
        logging.warning("No videos found")
        return False

    logging.info(f"Total videos: {len(video_paths)}")

    func = functools.partial(
        extract_frames_every_second,
        output_folder=output_folder
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        list(
            tqdm(
                executor.map(func, video_paths),
                total=len(video_paths),
                desc="Processing Videos"
            )
        )

    return True


def main():

    success = process_all_videos(max_workers=6)

    if not success:
        sys.exit(1)

    logging.info("Done")


if __name__ == "__main__":
    main()