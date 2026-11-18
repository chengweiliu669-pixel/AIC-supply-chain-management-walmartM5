import math
import torch
from torch import nn
from torch.nn import functional as F


def make_dates(count, device, weekly=False):
    day = torch.arange(count, device=device, dtype=torch.float32)
    elapsed = day * (7 if weekly else 1)
    cycles = [365.25, 30.4375] if weekly else [365.25, 30.4375, 7]
    angles = elapsed[:, None] * (math.tau / torch.tensor(cycles, device=device))
    return torch.cat((elapsed[:, None] / 365.25, angles.sin(), angles.cos()), dim=1)


class PatchTransformer(nn.Module):
    def __init__(self, date_dims, pairs, joint, tokens_per_patch=7, sales_history=False, patch_count=4, layers=None, covariate_dims=0):
        super().__init__()
        self.date = nn.Linear(date_dims, 32)
        self.pair = nn.Embedding(pairs, 32) if joint else None
        if joint:
            self.interaction = nn.Linear(32, 32)
        self.sales = nn.Linear(2, 32, bias=False) if sales_history else None
        self.covariates = nn.Linear(covariate_dims, 32, bias=False) if covariate_dims else None
        self.tokens_per_patch = tokens_per_patch
        width, heads = (128, 8) if sales_history else (64, 4)
        self.patch = nn.Linear(tokens_per_patch * 32, width)
        self.patch_count = patch_count
        self.position = nn.Parameter(torch.randn(patch_count, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, 0.05, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers or (8 if sales_history else 4), enable_nested_tensor=False)
        self.readout = nn.Sequential(nn.LayerNorm(width + 32), nn.Linear(width + 32, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, past_dates, forecast_date, pair, past_sales=None, patch_padding=None, past_covariates=None):
        history = self.date(past_dates)
        query = self.date(forecast_date)
        if self.pair is not None:
            key = self.pair(pair)
            history = history + key[:, None] + self.interaction(history * key[:, None])
            query = query + key + self.interaction(query * key)
        if self.sales is not None:
            observed = torch.isfinite(past_sales)
            values = torch.log1p(torch.where(observed, past_sales, 0).clamp_min(0))
            history = history + self.sales(torch.stack((values, observed.float()), dim=-1))
        if self.covariates is not None:
            history = history + self.covariates(past_covariates)
        tokens = self.patch(history.reshape(-1, self.patch_count, self.tokens_per_patch * 32))
        hidden = self.encoder(tokens + self.position, src_key_padding_mask=patch_padding)
        return F.softplus(self.readout(torch.cat((hidden[:, -1], query), dim=-1))).squeeze(-1)



