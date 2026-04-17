from dataclasses import dataclass

@dataclass
class TrainConfig:
    # data
    train_json: str = "/dshome/ddualab/jeehyeon/finetuning/ours/data/loo_mappings/mappings_train_re.json"
    val_json: str   = "/dshome/ddualab/jeehyeon/finetuning/ours/data/mapping_val.json"
    test_json: str  = "/dshome/ddualab/jeehyeon/finetuning/ours/data/loo_mappings/mappings_test_re.json"
    catalog_json: str = "/dshome/ddualab/jeehyeon/finetuning/ours/data/catalog_re.json"

    # dims & batching
    d: int = 512    
    batch_size: int = 128
    num_workers: int = 4

    # optimization
    lr: float = 3e-4
    epochs: int = 27
    weight_decay: float = 0.01
    tau: float = 0.03   
    # alpha: float = 0.1    
    # lam: float = 0.01    
    grad_clip: float = 1.0
    alpha_retr: float = 0.7
    lambda_gate_entropy: float = 0.005
    # cosine_margin: float = 0.05  
    logit_scale: float = 1.0      
    # inbatch_neg = 8
    # hard_neg = 16

    # misc
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "/dshome/ddualab/jeehyeon/finetuning/ours/checkpoints_train"
    log_every: int = 1
