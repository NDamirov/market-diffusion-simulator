from __future__ import annotations

import math

import numpy as np
import pandas as pd

from market_diffusion.config import ProjectConfig


def make_synthetic_prices(
    tickers: tuple[str, ...] | list[str],
    n_days: int = 2600,
    seed: int = 42,
) -> pd.DataFrame:
    """Offline fallback: correlated heavy-tailed returns converted to price paths."""
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    base_corr = 0.35 * np.ones((n_assets, n_assets)) + 0.65 * np.eye(n_assets)
    base_vols = np.array([0.23, 0.21, 0.38, 0.24, 0.27], dtype=float)
    vols = np.resize(base_vols, n_assets) / np.sqrt(252)
    cov = np.outer(vols, vols) * base_corr
    chol = np.linalg.cholesky(cov)

    returns = []
    sigma_state = np.ones(n_assets)
    for _ in range(n_days):
        shock = rng.standard_t(df=5, size=n_assets) / math.sqrt(5 / 3)
        eps = chol @ shock
        sigma_state = 0.96 * sigma_state + 0.04 * (0.6 + 8.0 * np.abs(eps) / vols)
        returns.append(0.00025 + eps * np.clip(sigma_state, 0.5, 3.0))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    prices = 100 * np.exp(np.cumsum(np.asarray(returns), axis=0))
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


def download_prices(cfg: ProjectConfig) -> pd.DataFrame:
    """Download adjusted close prices through yfinance, with an optional synthetic fallback."""
    try:
        import yfinance as yf

        raw = yf.download(
            list(cfg.tickers),
            start=cfg.start_date,
            end=cfg.end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print("Download failed:", repr(exc))
        raw = pd.DataFrame()

    if raw.empty:
        if not cfg.synthetic_fallback_if_download_fails:
            raise RuntimeError("No market data downloaded. Check internet access or yfinance.")
        print("Using synthetic fallback data. Re-run with internet for final project results.")
        return make_synthetic_prices(cfg.tickers, seed=cfg.seed)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError(f"Could not find close prices in columns: {raw.columns}")
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [cfg.tickers[0]]

    prices = prices.reindex(columns=list(cfg.tickers))
    return prices.dropna(how="all").ffill().dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff().dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def train_test_split_returns(
    returns: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(returns) * train_fraction)
    return returns.iloc[:split_idx].copy(), returns.iloc[split_idx:].copy()


def standardize_returns(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mean = train_returns.mean()
    std = train_returns.std().replace(0, np.nan)
    return (train_returns - mean) / std, (test_returns - mean) / std, mean, std
