import os
import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import PlaylistDataset, build_catalog_from_json
from utils import seed_all, ensure_dir, l2_normalize
from model_parts import RecommenderModel
from data_utils import collate_fn
from train_eval import run_epoch_onehot, compute_sampled_metrics, check_coverage



def main():
    cfg = TrainConfig()
    seed_all(cfg.seed)
    ensure_dir(cfg.save_dir)

    train_ds = PlaylistDataset(cfg.train_json, cfg.d)
    test_ds = PlaylistDataset(cfg.test_json, cfg.d)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = RecommenderModel(d=cfg.d).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"******Total params: {total_params:,}")
    print(f"******Trainable params: {trainable_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    catalog_ids, catalog_V = build_catalog_from_json(cfg.catalog_json, cfg.d)
    Vn = l2_normalize(catalog_V, dim=1)
    id2idx = {tid: i for i, tid in enumerate(catalog_ids)}

    check_coverage(train_loader, id2idx, "train")
    check_coverage(test_loader, id2idx, "test")

    eval_every = getattr(cfg, "eval_every", 1)

    for ep in range(1, cfg.epochs + 1):
        tr_loss = run_epoch_onehot(
            model, train_loader, Vn, id2idx, cfg, opt=opt, train=True
        )
        print(f"[{ep}] train CE={tr_loss:.4f}")

        if ep % eval_every == 0:
            sampled_test = compute_sampled_metrics(
                model, test_loader, catalog_ids, Vn, cfg,
                K_list=(10, 20, 50), n_neg=getattr(cfg, "n_neg_eval", 1000),
                remove_context=False, keep_target_in_candidates=True, seed=getattr(cfg, "seed", 42)
            )
            print(
                f"TEST(sampled N={getattr(cfg,'n_neg_eval',1000)}): "
                f"R@10={sampled_test['recall'][10]:.4f} | NDCG@10={sampled_test['ndcg'][10]:.4f} | "
                f"R@20={sampled_test['recall'][20]:.4f} | NDCG@20={sampled_test['ndcg'][20]:.4f} | "
                f"R@50={sampled_test['recall'][50]:.4f} | NDCG@50={sampled_test['ndcg'][50]:.4f} | "
                f"MRR={sampled_test['mrr']:.4f}"
            )

        if ep == cfg.epochs:
            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_dir, f"ep{ep:02d}_last_0105.pt")
            torch.save(
                {"ep": ep, "model": model.state_dict(), "cfg": vars(cfg)},
                ckpt_path
            )
            print(f"[ckpt] saved final model: {ckpt_path}")


if __name__ == "__main__":
    main()
