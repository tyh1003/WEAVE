import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

batch_size = 4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=batch_size,
    torch_dtype=torch_dtype,
    device=device,
)

class AudioDataset(Dataset):
    def __init__(self, vid_list, audio_dir):
        self.vid_list = vid_list
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid = self.vid_list[idx]
        return vid, os.path.join(self.audio_dir, f"{vid}.wav")

def process_batch(batch):
    vids, paths = batch
    vids = list(vids)
    paths = list(paths)

    results = {}
    valid_paths = []
    valid_vids = []

    for vid, path in zip(vids, paths):
        if os.path.exists(path):
            valid_paths.append(path)
            valid_vids.append(vid)
        else:
            results[vid] = ""

    if valid_paths:
        transcripts = pipe(valid_paths)
        for vid, out in zip(valid_vids, transcripts):
            results[vid] = out["text"]

    return [(vid, results.get(vid, "")) for vid in vids]

def process_dataset(dataset_name):
    label_file = f"data/{dataset_name}/label.jsonl"
    audio_dir = f"data/{dataset_name}/audios"
    dst_file = f"data/{dataset_name}/transcript.jsonl"

    if not os.path.exists(label_file):
        return

    df = pd.read_json(label_file, lines=True, dtype={'vid': str})
    vid_list = df['vid'].tolist()

    processed = set()
    if os.path.exists(dst_file):
        with open(dst_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["vid"])
                except:
                    pass

    vid_list = [v for v in vid_list if v not in processed]

    dataset = AudioDataset(vid_list, audio_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with open(dst_file, "a", encoding="utf-8") as out_file:
        for batch in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            results = process_batch(batch)
            for vid, transcript in results:
                record = {"vid": vid, "transcript": transcript}
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

    print(f"{dataset_name} complete")

if __name__ == "__main__":
    datasets = ["ALL"]
    for dataset_name in datasets:
        process_dataset(dataset_name)

    print("All datasets processed")
