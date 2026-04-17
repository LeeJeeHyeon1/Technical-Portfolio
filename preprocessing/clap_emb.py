import torch
import torchaudio
import numpy as np
import os
import glob
from tqdm import tqdm

from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2
from audioldm_train.modules.clap.training.data import get_audio_features

wav_root_dir = "/dshome/ddualab/jeehyeon/finetuning/AudioLDM-training-finetuning/audioldm_train/log/latent_diffusion/2023_08_23_reproduce_audioldm/infer_03-23-11:25_cfg_scale_3.5_ddim_50_n_cand_1"

output_root_dir = "/dshome/ddualab/jeehyeon/finetuning/ours/data/embeddings/gne_2000"
os.makedirs(output_root_dir, exist_ok=True)

clap_audio = CLAPAudioEmbeddingClassifierFreev2(
    pretrained_path="/dshome/ddualab/jeehyeon/finetuning/AudioLDM-training-finetuning/data/checkpoints/clap_htsat_tiny.pt",
    embed_mode="audio",
    amodel="HTSAT-tiny",
)

clap_audio = clap_audio.cuda()
clap_audio.eval()

print(f"✅ CLAP model loaded with sampling rate: {clap_audio.sampling_rate}")

target_length = clap_audio.sampling_rate * 10


def get_audio_embedding(clap_model, audio_tensor):
    with torch.no_grad():
        audio_tensor = audio_tensor.squeeze(1)  

        mel = clap_model.mel_transform(audio_tensor)

        audio_dict = get_audio_features(
            audio_tensor,
            mel,
            target_length,
            data_truncating="fusion",
            data_filling="repeatpad",
            audio_cfg=clap_model.model_cfg["audio_cfg"],
        )

        embed = clap_model.model.get_audio_embedding(audio_dict)
        return embed


def extract_clap_embedding(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != clap_audio.sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=clap_audio.sampling_rate
        )

    if waveform.shape[1] < target_length:
        pad_len = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    waveform = waveform.unsqueeze(0).cuda()  

    emb = get_audio_embedding(clap_audio, waveform)
    return emb.squeeze(0).cpu()


def parse_playlist_and_track_id(wav_path):
    filename = os.path.basename(wav_path)
    stem = os.path.splitext(filename)[0]

    if "_" not in stem:
        raise ValueError(f"Invalid filename format: {filename}")

    playlist_id, track_id = stem.rsplit("_", 1)
    return playlist_id, track_id


wav_files = glob.glob(os.path.join(wav_root_dir, "**", "*.wav"), recursive=True)
print(f"총 wav 파일 수: {len(wav_files)}")

saved_count = 0
failed_count = 0

for wav_path in tqdm(wav_files, desc="Extracting CLAP embeddings"):
    try:
        playlist_id, track_id = parse_playlist_and_track_id(wav_path)

        emb = extract_clap_embedding(wav_path)

        playlist_dir = os.path.join(output_root_dir, playlist_id)
        os.makedirs(playlist_dir, exist_ok=True)

        out_path = os.path.join(playlist_dir, f"{track_id}.npy")
        np.save(out_path, emb.numpy().astype("float32"))

        saved_count += 1

    except Exception as e:
        print(f"[ERROR] {wav_path} -> {repr(e)}")
        failed_count += 1

print(f"✅ Finished: {saved_count} embeddings saved")
print(f"❌ Failed: {failed_count}")