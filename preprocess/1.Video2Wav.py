import subprocess
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial


def convert_one(video_path, output_folder):
    """
    將單支影片轉成 wav
    """

    output_path = output_folder / f"{video_path.stem}.wav"

    # 已存在則跳過
    if output_path.exists():
        return None

    cmd = [
        'ffmpeg',

        '-y',

        '-i', str(video_path),

        # 不輸出影像
        '-vn',

        # mono
        '-ac', '1',

        # 16kHz
        '-ar', '16000',

        # wav codec
        '-acodec', 'pcm_s16le',

        # 單 ffmpeg process thread
        '-threads', '1',

        '-loglevel', 'error',

        str(output_path)
    ]

    try:

        subprocess.run(
            cmd,
            check=True
        )

        # 檢查檔案是否合法
        if not output_path.exists():
            return video_path.name

        if output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            return video_path.name

        return None

    except subprocess.CalledProcessError:

        return video_path.name


def convert_all_videos(max_workers=6):

    input_folder = Path('data/ALL/videos')

    output_folder = Path('data/ALL/audios')

    if not input_folder.exists():

        print(f"Input folder not found: {input_folder}")

        return

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # 支援格式
    extensions = [
        '*.mp4',
        '*.mov',
        '*.mkv',
        '*.avi'
    ]

    video_paths = []

    for ext in extensions:

        video_paths.extend(
            input_folder.glob(ext)
        )

    video_paths = sorted(video_paths)

    if len(video_paths) == 0:

        print("No videos found")

        return

    print(f"Total videos: {len(video_paths)}")

    func = partial(
        convert_one,
        output_folder=output_folder
    )

    # multiprocessing
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        results = list(
            tqdm(
                executor.map(func, video_paths),
                total=len(video_paths),
                desc="Converting Videos"
            )
        )

    # 收集失敗影片
    errors = [r for r in results if r]

    if errors:

        print("\nFailed files:")

        for e in errors:
            print(e)

    else:

        print("\nAll conversions complete!")


if __name__ == "__main__":

    convert_all_videos(max_workers=6)

    print("\nDone!")