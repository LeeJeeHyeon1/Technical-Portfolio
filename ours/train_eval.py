import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import _dbg_write, gate_entropy_loss
from sampling import simple_sample_candidates, _dcg_from_rank



def run_epoch_onehot(model, loader, Vn, id2idx, cfg, opt=None, train=True):
    device = cfg.device
    model.train() if train else model.eval()

    Vn = Vn.to(device)
    C, d = Vn.shape

    alpha_retr = float(getattr(cfg, "alpha_retr", 0.0))
    lambda_ge = float(getattr(cfg, "lambda_gate_entropy", 0.0))
    logit_scale = float(getattr(cfg, "logit_scale", 1.0))

    tot_loss, tot_cnt = 0.0, 0
    any_valid = False

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        e_a = batch["e_a"].to(device)
        e_t = batch["e_t"].to(device)
        attn = batch["attn_mask"].to(device)
        pos_ids = batch["pos_ids"]
        e_pos_all = batch["e_pos"].to(device)

        targets = torch.tensor([id2idx.get(tid, -1) for tid in pos_ids], device=device, dtype=torch.long)
        valid_mask = targets.ge(0)
        if not valid_mask.any():
            continue
        any_valid = True

        with torch.set_grad_enabled(train):
            z, g, tau, _ = model(e_t, e_a, attn)
            qv = z[valid_mask]
            tv = targets[valid_mask]
            e_pos_v = F.normalize(e_pos_all[valid_mask], dim=-1)

            logits = (qv @ Vn.t()) / tau
            if logit_scale != 1.0:
                logits = logits * logit_scale

            ce_cat = F.cross_entropy(logits, tv)

            loss = ce_cat

            if alpha_retr > 0.0:
                retr_logits = (qv @ e_pos_v.t()) / tau
                if logit_scale != 1.0:
                    retr_logits = retr_logits * logit_scale
                retr_labels = torch.arange(qv.size(0), device=device)
                ce_retr = F.cross_entropy(retr_logits, retr_labels)
                loss = loss + alpha_retr * ce_retr
            else:
                ce_retr = torch.tensor(0.0, device=device)

            if lambda_ge > 0.0:
                loss = loss + lambda_ge * gate_entropy_loss(g)

            if train and getattr(cfg, "dbg", True):
                with torch.no_grad():
                    V_pos_dbg = Vn.index_select(0, tv)
                    cos_q_vpos = (qv * V_pos_dbg).sum(-1).mean().item()
                    cos_q_epos = (qv * e_pos_v).sum(-1).mean().item()
                    tval = float(tau.item()) if torch.is_tensor(tau) else float(tau)
                    _dbg_write(f"[dbg] tau={tval:.4f} | g mean/std={g.mean().item():.3f}/{g.std().item():.3f} | cos(q,V_pos)={cos_q_vpos:.3f} | cos(q,e_pos)={cos_q_epos:.3f} | CE(cat)={float(ce_cat.item()):.3f} | CE(retr)={float(ce_retr.item()):.3f}")

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if getattr(cfg, "grad_clip", None):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

        bsz = qv.size(0)
        tot_loss += float(loss.item()) * bsz
        tot_cnt += bsz

    if not any_valid:
        print("[WARN] No valid targets found this epoch. (catalog mismatch?)")

    return tot_loss / max(tot_cnt, 1)


@torch.no_grad()
def compute_full_recall(model, loader, catalog_ids, Vn, cfg, K_list=(10, 20, 50), exclude_history=True):
    tid2indices = {}
    for i, tid in enumerate(catalog_ids):
        tid2indices.setdefault(tid, []).append(i)

    Vn = Vn.to(cfg.device)
    hits = {K: 0 for K in K_list}
    totals = 0

    for batch in loader:
        e_a = batch["e_a"].to(cfg.device)
        e_t = batch["e_t"].to(cfg.device)
        attn_mask = batch["attn_mask"].to(cfg.device)
        pos_ids = batch["pos_ids"]
        ctx_ids_lists = batch.get("ctx_ids", [[] for _ in pos_ids])

        out = model(e_t, e_a, attn_mask)
        q = out[0] if isinstance(out, tuple) else out
        sims = q @ Vn.t()

        for r, (tid, ctx_ids) in enumerate(zip(pos_ids, ctx_ids_lists)):
            if tid not in tid2indices:
                continue

            if exclude_history and ctx_ids:
                mask_indices = []
                for cid in ctx_ids:
                    if cid == tid:
                        continue
                    mask_indices.extend(tid2indices.get(cid, []))
                if mask_indices:
                    sims[r, torch.tensor(mask_indices, device=sims.device, dtype=torch.long)] = float("-inf")

            totals += 1
            tgt_indices = set(tid2indices[tid])

            for K in K_list:
                topk = torch.topk(sims[r], k=min(K, Vn.size(0))).indices.tolist()
                if any(idx in tgt_indices for idx in topk):
                    hits[K] += 1

    return {K: (hits[K] / totals if totals > 0 else 0.0) for K in K_list}


@torch.no_grad()
def compute_sampled_metrics(
    model, loader, catalog_ids, Vn, cfg,
    K_list=(10, 20, 50),
    n_neg: int = 1000,
    remove_context: bool = True,
    keep_target_in_candidates: bool = True,
    seed: int = 1234
):
    was_training = model.training
    model.eval()
    try:
        id2idx = {tid: i for i, tid in enumerate(catalog_ids)}
        Vn = Vn.to(cfg.device)

        hits = {K: 0 for K in K_list}
        ndcgs = {K: 0.0 for K in K_list}
        mrr_sum = 0.0
        totals = 0

        g = torch.Generator(device=cfg.device)
        g.manual_seed(seed)

        for batch in loader:
            e_t = batch["e_t"].to(cfg.device)
            e_a = batch["e_a"].to(cfg.device)
            attn_mask = batch["attn_mask"].to(cfg.device)
            pos_ids = batch["pos_ids"]
            ctx_lists = batch.get("ctx_ids", [[] for _ in pos_ids]) if remove_context else [[] for _ in pos_ids]

            out = model(e_t, e_a, attn_mask)
            q = out[0] if isinstance(out, tuple) else out
            q = F.normalize(q, dim=-1)

            cand_idx, pos_col = simple_sample_candidates(
                id2idx=id2idx,
                pos_ids=pos_ids,
                n_items=Vn.size(0),
                n_neg=n_neg,
                ctx_lists=ctx_lists,
                keep_target_in_candidates=keep_target_in_candidates,
                device=cfg.device,
                rng=g
            )

            B = q.size(0)
            for r in range(B):
                tgt_idx = id2idx.get(pos_ids[r], None)
                if tgt_idx is None:
                    continue

                cidx = cand_idx[r]
                if cidx.numel() == 0:
                    continue

                Vc = Vn.index_select(0, cidx)
                sims = (q[r].unsqueeze(0) @ Vc.t()).squeeze(0)

                if keep_target_in_candidates:
                    gt_col = int(pos_col[r].item())
                    if not (0 <= gt_col < sims.numel()):
                        continue
                else:
                    continue

                totals += 1
                ranks_desc = torch.argsort(sims, descending=True)
                where = (ranks_desc == gt_col).nonzero(as_tuple=False)
                best_rank = int(where[0].item()) + 1 if where.numel() > 0 else None

                for K in K_list:
                    k = min(K, sims.numel())
                    topk = ranks_desc[:k]
                    if (topk == gt_col).any().item():
                        hits[K] += 1
                    ndcgs[K] += _dcg_from_rank(best_rank) if (best_rank is not None and best_rank <= K) else 0.0

                if best_rank is not None:
                    mrr_sum += 1.0 / float(best_rank)

        recalls = {K: (hits[K] / totals if totals > 0 else 0.0) for K in K_list}
        ndcg = {K: (ndcgs[K] / totals if totals > 0 else 0.0) for K in K_list}
        mrr = (mrr_sum / totals) if totals > 0 else 0.0
        return {"recall": recalls, "ndcg": ndcg, "mrr": mrr}
    finally:
        if was_training:
            model.train()



def check_coverage(loader, id2idx, name="loader"):
    covered = missing = 0
    for batch in loader:
        for tid in batch["pos_ids"]:
            if tid in id2idx:
                covered += 1
            else:
                missing += 1
    tot = covered + missing
    rate = (covered / tot) if tot else 0.0
    print(f"[COVERAGE:{name}] {covered}/{tot} = {rate:.2%} (missing={missing})")
