import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Dict

class PlaylistDataset(Dataset):
    def __init__(self, json_path: str, d: int):
        self.d = d
        with open(json_path, "r") as f:
            items = json.load(f)

        self.pl_map: Dict[str, List[Dict]] = defaultdict(list)
        for it in items:
            self.pl_map[str(it["playlist_id"])].append(it)

        self.playlists = []
        for pid, seq in self.pl_map.items():
            if len(seq) >= 2:
                self.playlists.append((pid, seq))

    def __len__(self):
        return len(self.playlists)

    def _load_vec(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 1 or arr.shape[0] != self.d:
            raise ValueError(f"Embedding shape mismatch at {path}: {arr.shape} expected ({self.d},)")
        return arr.astype(np.float32)

    def __getitem__(self, idx: int):
        pid, seq = self.playlists[idx]
        ctx = seq[:-1]
        tgt = seq[-1]

        e_a_ctx = np.stack([self._load_vec(x["audio_path"]) for x in ctx], axis=0)
        e_t_ctx = np.stack([self._load_vec(x["gen_path"])   for x in ctx], axis=0)
        
        e_pos = self._load_vec(tgt["audio_path"])

        return {
            "e_a": torch.from_numpy(e_a_ctx),  
            "e_t": torch.from_numpy(e_t_ctx),  
            "e_pos": torch.from_numpy(e_pos),   
            "pid": str(pid),
            "pos_id": str(tgt["track_id"]),    
            "ctx_ids": [str(x["track_id"]) for x in ctx],
            "len": e_a_ctx.shape[0]
        }
        
        

def build_catalog_from_json(catalog_json: str, d: int):
    with open(catalog_json, "r") as f:
        items = json.load(f)
    ids, vecs = [], []
    for it in items:
        ids.append(str(it["track_id"]))
        v = np.load(it["audio_path"]).astype(np.float32)
        if v.ndim != 1 or v.shape[0] != d:
            raise ValueError(f"Catalog embedding shape mismatch: {v.shape}, expected ({d},)")
        vecs.append(v)
    V = torch.from_numpy(np.stack(vecs, axis=0))  # (N,d)
    print('All catalog:', len(ids))
    return ids, V
