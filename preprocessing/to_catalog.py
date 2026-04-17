import os
import json

def build_simple_metadata(audio_root, output_json_path):
    metadata = []

    for file_name in os.listdir(audio_root):
        if not file_name.endswith(".npy"):
            continue

        track_id = file_name.replace("wav_", "").replace(".npy", "")

        audio_path = os.path.join(audio_root, file_name)

        metadata.append({
            "track_id": track_id,
            "audio_path": audio_path
        })

    with open(output_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_json_path}")

# 사용 예시
audio_root = "/dshome/ddualab/jeehyeon/finetuning/ours/data/embeddings/audio"
output_json_path = "/dshome/ddualab/jeehyeon/finetuning/ours/data/metadata_audio.json"

build_simple_metadata(audio_root, output_json_path)