import os
import pandas as pd
import json
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

BASE_DIR = "./data/FVC"
os.makedirs(os.path.join(BASE_DIR, "videos"), exist_ok=True)

df = pd.read_csv(os.path.join(BASE_DIR, "FVC.csv"))
data_list = []
label_list = []
fail_log = []

def extract_vid(url):
    parsed = urlparse(url)
    if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
        # YouTube
        query = parse_qs(parsed.query)
        if 'v' in query:
            return query["v"][0], 'youtube'
        elif 'youtu.be' in parsed.netloc:
            return parsed.path.strip('/'), 'youtube'
    elif 'facebook.com' in parsed.netloc:
        # Facebook
        # 通常格式 /videos/123456789/ or /watch?v=123456789
        try:
            parts = parsed.path.split('/')
            vid = [p for p in parts if p.isdigit()]
            if vid:
                return vid[0], 'facebook'
        except:
            pass
    # 其他平台（可加更多）
    # 最保險直接對 url 做 hash
    return hashlib.md5(url.encode()).hexdigest(), 'other'

def download_video(row):
    video_url = row['video_url']
    label_text = row['label'].strip().lower()
    label_num = 1 if label_text == 'fake' else 0
    vid, platform = extract_vid(video_url)

    output_path = os.path.join(BASE_DIR, "videos", f"{vid}.mp4")
    if os.path.exists(output_path):
        return vid, label_text, label_num, platform, f"⏩ 已存在，跳過：{vid}", None

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'merge_output_format': 'mp4',
        'outtmpl': output_path,
        'cookiefile': os.path.join(BASE_DIR, 'www.youtube.com_cookies.txt')
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return vid, label_text, label_num, platform, f"✅ 下載成功：{vid}", None
    except Exception as e:
        return None, None, None, platform, f"❌ 下載失敗 {video_url}：{e}", video_url

print("🚀 開始多執行緒下載...")
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(download_video, row) for _, row in df.iterrows()]
    for future in as_completed(futures):
        vid, label_str, label_num, platform, msg, fail_url = future.result()
        print(msg)
        if vid is not None:
            data_list.append({"vid": vid, "label": label_str, "platform": platform})
            label_list.append({"vid": vid, "label": label_num})
        elif fail_url:
            fail_log.append(fail_url)

    # 照原本 query 檔補 title...
    queries_df = pd.read_csv(os.path.join(BASE_DIR, "FVC_text_queries.csv"))
    vid2title = {}
    for _, row in queries_df.iterrows():
        v, _ = extract_vid(row['video_url'])
        vid2title[v] = row['event_title']
    for entry in data_list:
        entry["title"] = vid2title.get(entry["vid"], "")

with open(os.path.join(BASE_DIR, "data.jsonl"), "w", encoding="utf-8") as f:
    for entry in data_list:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")
with open(os.path.join(BASE_DIR, "label.jsonl"), "w", encoding="utf-8") as f:
    for entry in label_list:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")
if fail_log:
    print(f"\n⚠️ 有 {len(fail_log)} 部影片下載失敗：")
    for url in fail_log:
        print(url)
else:
    print("\n✅ 所有影片下載成功，無失敗影片")

print("🎉 JSONL 檔案與影片處理完畢")
