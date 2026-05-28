from __future__ import annotations

import numpy as np
import pandas as pd


def plot_price_and_returns(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    (prices / prices.iloc[0]).plot(ax=ax, linewidth=1.5)
    ax.set_title("Normalized adjusted close prices")
    ax.set_ylabel("Price / initial price")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    returns.plot(ax=ax, linewidth=0.6, alpha=0.75)
    ax.set_title("Daily log-returns")
    ax.set_ylabel("log-return")
    plt.show()


def plot_training_loss(history: list[float], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE noise prediction loss")
    plt.show()


def plot_return_distribution(data_by_name: dict[str, np.ndarray], asset_idx: int, asset: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, arr in data_by_name.items():
        sns.histplot(
            arr[:, asset_idx],
            bins=80,
            stat="density",
            element="step",
            fill=False,
            linewidth=1.6,
            label=label,
            ax=ax,
        )
    ax.set_title(f"Daily return distribution: {asset}")
    ax.set_xlabel("daily log-return")
    ax.legend()
    plt.show()


def plot_correlation_heatmaps(correlations: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, len(correlations), figsize=(5.4 * len(correlations), 4.5))
    if len(correlations) == 1:
        axes = [axes]
    for ax, (title, corr) in zip(axes, correlations.items()):
        sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", square=True, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def windows_to_price_paths(
    windows: np.ndarray,
    asset_idx: int,
    s0: float,
    n_paths: int = 80,
    seed: int = 42,
) -> np.ndarray:
    n = min(n_paths, len(windows))
    idx = np.random.default_rng(seed).choice(len(windows), size=n, replace=False)
    returns_selected = windows[idx, :, asset_idx]
    paths = s0 * np.exp(np.cumsum(returns_selected, axis=1))
    return np.concatenate([np.full((n, 1), s0), paths], axis=1)


def plot_price_paths_by_source(
    windows_by_name: dict[str, np.ndarray],
    asset_idx: int,
    asset: str,
    s0: float,
    n_paths: int = 60,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(windows_by_name), figsize=(5.4 * len(windows_by_name), 4), sharey=True)
    if len(windows_by_name) == 1:
        axes = [axes]
    for ax, (title, windows) in zip(axes, windows_by_name.items()):
        paths = windows_to_price_paths(windows, asset_idx, s0, n_paths=n_paths)
        ax.plot(paths.T, color="tab:blue", alpha=0.12, linewidth=1)
        ax.plot(paths.mean(axis=0), color="black", linewidth=2, label="mean path")
        ax.set_title(f"{title}: {asset}")
        ax.set_xlabel("trading day")
        ax.legend()
    axes[0].set_ylabel("synthetic price")
    plt.tight_layout()
    plt.show()


def plot_conditional_regime_distributions(
    conditional_windows_by_regime: dict[str, np.ndarray],
    vol_fn,
    horizon_return_fn,
    thresholds: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    low_q, high_q = thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for regime_name, windows in conditional_windows_by_regime.items():
        sns.kdeplot(vol_fn(windows), label=regime_name, linewidth=2, ax=axes[0])
        sns.kdeplot(horizon_return_fn(windows), label=regime_name, linewidth=2, ax=axes[1])

    axes[0].axvline(low_q, color="tab:orange", linestyle="--", linewidth=1)
    axes[0].axvline(high_q, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_title("Generated realized volatility by requested regime")
    axes[0].set_xlabel("annualized realized volatility")
    axes[1].axvline(0, color="black", linestyle=":", linewidth=1)
    axes[1].set_title("Generated 20-day portfolio log-return by requested regime")
    axes[1].set_xlabel("20-day equal-weight portfolio log-return")
    for ax in axes:
        ax.legend(title="requested regime")
    plt.tight_layout()
    plt.show()
