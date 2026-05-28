from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProjectConfig:
    tickers: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-31"
    train_fraction: float = 0.80
    window_size: int = 20
    diffusion_steps: int = 100
    batch_size: int = 256
    epochs_gpu: int = 500
    epochs_cpu: int = 250
    conditional_epochs_gpu: int = 250
    conditional_epochs_cpu: int = 120
    learning_rate: float = 2e-4
    hidden_dim: int = 256
    time_emb_dim: int = 64
    regime_emb_dim: int = 32
    n_scenarios: int = 3000
    conditional_n_per_regime: int = 1000
    guidance_scale: float = 1.5
    conditional_label_drop_prob: float = 0.15
    clip_daily_return: float = 0.35
    portfolio_alpha: float = 0.05
    option_asset: str = "AAPL"
    option_maturity_days: int = 30
    option_moneyness: float = 1.00
    risk_free_rate: float = 0.04
    synthetic_fallback_if_download_fails: bool = True
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
