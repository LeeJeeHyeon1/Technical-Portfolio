import math
import torch
import torch.nn.functional as F



def simple_sample_candidates(
    id2idx: dict,
    pos_ids: list,
    n_items: int,
    n_neg: int,
    ctx_lists=None,
    keep_target_in_candidates: bool = True,
    device: str = "cpu",
    seed: int = None,
    rng: torch.Generator = None
):
    B = len(pos_ids)
    M = n_neg + (1 if keep_target_in_candidates else 0)

    if rng is not None:
        g = rng
    elif seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    cand_idx = torch.empty((B, M), dtype=torch.long, device=device)
    pos_col = torch.full((B,), (n_neg if keep_target_in_candidates else -1), dtype=torch.long, device=device)

    all_idx = torch.arange(n_items, device=device)

    if ctx_lists is None:
        ctx_lists = [[] for _ in range(B)]

    for r, tid in enumerate(pos_ids):
        tgt = id2idx.get(tid, None)
        if tgt is None:
            cand_idx[r].fill_(0)
            pos_col[r] = -1
            continue

        mask = torch.ones(n_items, dtype=torch.bool, device=device)
        mask[tgt] = False
        for c in ctx_lists[r]:
            if c in id2idx:
                mask[id2idx[c]] = False

        allow = all_idx[mask]

        if allow.numel() == 0:
            allow = all_idx[all_idx != tgt]

        if allow.numel() == 0:
            neg = torch.full((n_neg,), tgt, dtype=torch.long, device=device)
        else:
            idx = torch.randint(0, allow.numel(), (n_neg,), generator=g, device=device)
            neg = allow.index_select(0, idx)

        if keep_target_in_candidates:
            cand_idx[r, :n_neg] = neg
            cand_idx[r, n_neg] = tgt
        else:
            cand_idx[r, :n_neg] = neg

    return cand_idx, pos_col



def _dcg_from_rank(rank: int):
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1)



def _best_rank_among(indices_sorted_desc, relevant_set):
    for r, idx in enumerate(indices_sorted_desc.tolist(), start=1):
        if idx in relevant_set:
            return r
    return None
