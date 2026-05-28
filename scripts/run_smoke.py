from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from market_diffusion.config import ProjectConfig, get_device, seed_everything
from market_diffusion.data import compute_log_returns, make_synthetic_prices, standardize_returns, train_test_split_returns
from market_diffusion.diffusion import (
    ConditionalDenoiseMLP,
    DenoiseMLP,
    DiffusionSchedule,
    sample_conditional_ddpm,
    sample_ddpm,
    train_conditional_ddpm,
    train_ddpm,
)
from market_diffusion.windows import (
    REGIME_NAMES,
    build_regime_labels,
    inverse_scale_windows,
    make_windows,
)


def main() -> None:
    cfg = ProjectConfig(
        window_size=10,
        diffusion_steps=5,
        batch_size=64,
        epochs_cpu=1,
        epochs_gpu=1,
        conditional_epochs_cpu=1,
        conditional_epochs_gpu=1,
        hidden_dim=64,
        n_scenarios=16,
        conditional_n_per_regime=8,
    )
    seed_everything(cfg.seed)
    device = get_device()

    prices = make_synthetic_prices(cfg.tickers, n_days=350, seed=cfg.seed)
    returns = compute_log_returns(prices)
    train_returns, test_returns = train_test_split_returns(returns, cfg.train_fraction)
    train_scaled, test_scaled, train_mean, train_std = standardize_returns(train_returns, test_returns)

    train_windows_scaled = make_windows(train_scaled, cfg.window_size)
    train_windows_raw = make_windows(train_returns, cfg.window_size)
    n_assets = len(cfg.tickers)
    input_dim = cfg.window_size * n_assets

    schedule = DiffusionSchedule(cfg.diffusion_steps, device)
    model = DenoiseMLP(input_dim, hidden_dim=cfg.hidden_dim, time_emb_dim=cfg.time_emb_dim).to(device)
    train_ddpm(
        model,
        schedule,
        train_windows_scaled,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        device=device,
        epochs=1,
    )
    samples_scaled = sample_ddpm(
        model,
        schedule,
        cfg.n_scenarios,
        input_dim,
        cfg.window_size,
        n_assets,
        device,
    )
    samples = inverse_scale_windows(samples_scaled, train_mean, train_std, cfg.clip_daily_return)
    assert samples.shape == (cfg.n_scenarios, cfg.window_size, n_assets)

    regime_labels, _, _ = build_regime_labels(train_windows_raw)
    n_regimes = len(REGIME_NAMES)
    unknown = n_regimes
    cond_model = ConditionalDenoiseMLP(
        input_dim,
        hidden_dim=cfg.hidden_dim,
        time_emb_dim=cfg.time_emb_dim,
        n_regimes=n_regimes,
        regime_emb_dim=cfg.regime_emb_dim,
    ).to(device)
    train_conditional_ddpm(
        cond_model,
        schedule,
        train_windows_scaled,
        regime_labels,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        device=device,
        epochs=1,
        unknown_regime_label=unknown,
    )
    conditional_scaled = sample_conditional_ddpm(
        cond_model,
        schedule,
        cfg.conditional_n_per_regime,
        input_dim,
        cfg.window_size,
        n_assets,
        device,
        regime_label=2,
        unknown_regime_label=unknown,
        guidance_scale=cfg.guidance_scale,
    )
    conditional = inverse_scale_windows(conditional_scaled, train_mean, train_std, cfg.clip_daily_return)
    assert conditional.shape == (cfg.conditional_n_per_regime, cfg.window_size, n_assets)

    print("Smoke test OK")
    print("unconditional sample mean:", np.round(samples.mean(), 6))
    print("conditional stress sample mean:", np.round(conditional.mean(), 6))


if __name__ == "__main__":
    main()
