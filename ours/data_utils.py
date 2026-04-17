import torch
from tqdm import tqdm


def _dbg_write(msg: str):
    try:
        tqdm.write(msg)
    except Exception:
        print(msg, flush=True)



def collate_fn(batch):
    d = batch[0]["e_a"].shape[1]
    lens = [b["len"] for b in batch]
    maxL = max(lens)

    def pad_stack(key):
        out = []
        for b in batch:
            x = b[key]
            pad = maxL - x.shape[0]
            if pad > 0:
                x = torch.cat([x, torch.zeros(pad, d, dtype=x.dtype)], dim=0)
            out.append(x)
        return torch.stack(out, dim=0)

    e_a = pad_stack("e_a")
    e_t = pad_stack("e_t")
    e_pos = torch.stack([b["e_pos"] for b in batch], dim=0)
    ctx_ids = [b["ctx_ids"] for b in batch]

    attn_mask = torch.ones((len(batch), maxL), dtype=torch.bool)
    for i, L in enumerate(lens):
        attn_mask[i, :L] = False

    pos_ids = [b["pos_id"] for b in batch]
    pids = [b["pid"] for b in batch]

    return {
        "e_a": e_a,
        "e_t": e_t,
        "e_pos": e_pos,
        "attn_mask": attn_mask,
        "lens": torch.tensor(lens),
        "pos_ids": pos_ids,
        "ctx_ids": ctx_ids,
        "pids": pids,
    }



def gate_entropy_loss(g, eps=1e-6):
    p = g.clamp(eps, 1 - eps)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return -ent.mean()
