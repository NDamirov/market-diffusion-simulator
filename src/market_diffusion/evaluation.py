from __future__ import annotations

import numpy as np
import pandas as pd


def stats_table(arr: np.ndarray, name: str, tickers: tuple[str, ...] | list[str]) -> pd.DataFrame:
    df = pd.DataFrame(arr, columns=list(tickers))
    return pd.DataFrame(
        {
            "dataset": name,
            "asset": list(tickers),
            "mean": df.mean().values,
            "std": df.std().values,
            "skew": df.skew().values,
            "excess_kurtosis": df.kurtosis().values,
            "q01": df.quantile(0.01).values,
            "q05": df.quantile(0.05).values,
            "q95": df.quantile(0.95).values,
            "q99": df.quantile(0.99).values,
        }
    )


def corr_df(arr: np.ndarray, tickers: tuple[str, ...] | list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.corrcoef(arr, rowvar=False), index=list(tickers), columns=list(tickers))


def correlation_error(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    return float(np.linalg.norm(candidate.values - reference.values, ord="fro"))


def var_cvar_from_returns(
    asset_returns: np.ndarray,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    if weights is None:
        weights = np.ones(asset_returns.shape[1]) / asset_returns.shape[1]
    portfolio_returns = asset_returns @ weights
    losses = -portfolio_returns
    var = np.quantile(losses, 1 - alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def scenario_var_cvar(window_log_returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    losses = -np.asarray(window_log_returns)
    var = np.quantile(losses, 1 - alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def average_window_acf(
    windows: np.ndarray,
    asset_idx: int,
    max_lag: int = 10,
    squared: bool = False,
) -> np.ndarray:
    values = windows[:, :, asset_idx]
    if squared:
        values = values**2
    acfs = []
    for lag in range(1, max_lag + 1):
        per_window = []
        for row in values:
            x = row[:-lag]
            y = row[lag:]
            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                per_window.append(np.corrcoef(x, y)[0, 1])
        acfs.append(np.nanmean(per_window))
    return np.asarray(acfs)


def sample_gaussian_windows(
    train_windows_scaled: np.ndarray,
    n_samples: int,
    jitter: float = 1e-5,
) -> np.ndarray:
    """Sample a multivariate Gaussian baseline fitted to flattened return windows."""
    flat = train_windows_scaled.reshape(len(train_windows_scaled), -1)
    mu = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 0:
        cov = cov + (-min_eig + jitter) * np.eye(cov.shape[0])
    else:
        cov = cov + jitter * np.eye(cov.shape[0])
    samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    return samples.reshape(n_samples, train_windows_scaled.shape[1], train_windows_scaled.shape[2])
