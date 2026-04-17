import os
import json

def build_metadata(embedding_root, output_json_path):
    metadata = []

    for playlist_id in os.listdir(embedding_root):
        playlist_path = os.path.join(embedding_root, playlist_id)

        if not os.path.isdir(playlist_path):
            continue

        for file_name in os.listdir(playlist_path):
            if not file_name.endswith(".npy"):
                continue

            track_id = os.path.splitext(file_name)[0]

            audio_path = os.path.join(playlist_path, file_name)

            gen_file_name = f"{track_id}_gen.npy"
            gen_path = os.path.join(playlist_path, gen_file_name)

            if not os.path.exists(gen_path):
                gen_path = None

            metadata.append({
                "playlist_id": playlist_id,
                "track_id": track_id,
                "audio_path": audio_path,
                "gen_path": gen_path
            })

    with open(output_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_json_path}")


# 사용 예시
embedding_root = "/path/to/embeddings"
output_json_path = "/path/to/metadata.json"

build_metadata(embedding_root, output_json_path)