import torch
import torch.nn as nn
import torch.nn.functional as F


class PerStepGate(nn.Module):
    def __init__(self, d: int, hidden: int = None):
        super().__init__()
        h = hidden or d
        self.net = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.GELU(),
            nn.Linear(h, 1)
        )

    def forward(self, e_t, e_a):
        x = torch.cat([e_t, e_a], dim=-1)
        return torch.sigmoid(self.net(x))


def build_causal_mask(L: int, device=None, dtype=torch.bool):
    m = torch.triu(torch.ones(L, L, dtype=dtype, device=device), diagonal=1)
    return m.bool()


class RecommenderModel(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        gate_hidden: int = None,
        max_len: int = 8192,
        use_input_ln: bool = True
    ):
        super().__init__()
        self.d = d
        self.gate = PerStepGate(d, hidden=gate_hidden)

        self.use_input_ln = use_input_ln
        self.in_ln = nn.LayerNorm(d) if use_input_ln else nn.Identity()
        self.in_proj = nn.Linear(d, d, bias=False)

        self.pos_emb = nn.Embedding(max_len, d)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        self.pe_scale = nn.Parameter(torch.tensor(0.10))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
        )

        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, e_t, e_a, pad_mask=None):
        g = self.gate(e_t, e_a)
        h = (1 - g) * e_t + g * e_a

        h = self.in_proj(self.in_ln(h))

        B, L, d = h.shape
        if L > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {L} exceeds max_len={self.pos_emb.num_embeddings}.")
        pos_ids = torch.arange(L, device=h.device).unsqueeze(0).expand(B, L)
        h = h + self.pe_scale * self.pos_emb(pos_ids)

        causal = build_causal_mask(L, device=h.device)
        out = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)

        if pad_mask is None:
            last_idx = torch.full((B,), L - 1, device=h.device, dtype=torch.long)
        else:
            valid_len = (~pad_mask).sum(dim=1)
            last_idx = (valid_len - 1).clamp(min=0)

        q = out[torch.arange(B, device=h.device), last_idx]

        z_pre = self.head(q)
        z_pre = z_pre + q
        z = F.normalize(z_pre, dim=-1)

        tau = torch.exp(self.log_tau).clamp(1e-2, 0.5)
        return z, g, tau, z_pre
