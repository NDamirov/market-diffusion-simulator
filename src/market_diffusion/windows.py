from __future__ import annotations

import math

import numpy as np
import pandas as pd


REGIME_NAMES = {0: "calm", 1: "normal", 2: "stress"}


def make_windows(df: pd.DataFrame, window_size: int) -> np.ndarray:
    arr = df.to_numpy(dtype=np.float32)
    if len(arr) < window_size:
        raise ValueError("Not enough rows to create rolling windows.")
    return np.stack([arr[i : i + window_size] for i in range(len(arr) - window_size + 1)])


def inverse_scale_windows(
    windows_scaled: np.ndarray,
    train_mean: pd.Series,
    train_std: pd.Series,
    clip_daily_return: float | None = None,
) -> np.ndarray:
    mean = train_mean.to_numpy(dtype=np.float32).reshape(1, 1, -1)
    std = train_std.to_numpy(dtype=np.float32).reshape(1, 1, -1)
    windows = windows_scaled * std + mean
    if clip_daily_return is not None:
        windows = np.clip(windows, -clip_daily_return, clip_daily_return)
    return windows


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(-1, windows.shape[-1])


def window_realized_volatility(windows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(windows.shape[2]) / windows.shape[2]
    portfolio_returns = windows @ weights
    return portfolio_returns.std(axis=1, ddof=1) * math.sqrt(252)


def window_portfolio_log_return(windows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(windows.shape[2]) / windows.shape[2]
    return (windows @ weights).sum(axis=1)


def assign_vol_regimes(
    volatility: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    labels = np.zeros(len(volatility), dtype=np.int64)
    labels[volatility > low_threshold] = 1
    labels[volatility > high_threshold] = 2
    return labels


def build_regime_labels(
    train_windows_raw: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    train_window_vol = window_realized_volatility(train_windows_raw)
    low_q, high_q = np.quantile(train_window_vol, [1 / 3, 2 / 3])
    labels = assign_vol_regimes(train_window_vol, low_q, high_q)
    return labels, (float(low_q), float(high_q)), train_window_vol


def regime_reference_table(
    labels: np.ndarray,
    thresholds: tuple[float, float],
    regime_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    if regime_names is None:
        regime_names = REGIME_NAMES
    low_q, high_q = thresholds
    return pd.DataFrame(
        {
            "regime": [regime_names[i] for i in range(len(regime_names))],
            "train_window_count": [int((labels == i).sum()) for i in range(len(regime_names))],
            "volatility_rule": [
                f"vol <= {low_q:.3f}",
                f"{low_q:.3f} < vol <= {high_q:.3f}",
                f"vol > {high_q:.3f}",
            ],
        }
    )
