from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_call_price_delta(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    S_arr = np.asarray(S, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    T = max(float(T), 1e-6)
    sqrt_T = math.sqrt(T)
    d1 = (np.log(np.maximum(S_arr, 1e-12) / K) + (r + 0.5 * sigma_arr**2) * T) / (
        sigma_arr * sqrt_T
    )
    d2 = d1 - sigma_arr * sqrt_T
    price = S_arr * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta


def realized_vol(window_returns: np.ndarray) -> np.ndarray:
    sigma = np.std(window_returns, axis=1, ddof=1) * math.sqrt(252)
    return np.clip(sigma, 0.01, 3.0)


def option_scenario_table(
    windows: np.ndarray,
    name: str,
    asset: str,
    asset_idx: int,
    prices: pd.DataFrame,
    train_returns: pd.DataFrame,
    maturity_days: int,
    window_size: int,
    moneyness: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    asset_windows = windows[:, :, asset_idx]
    T0 = maturity_days / 252
    T1 = max((maturity_days - window_size) / 252, 1 / 252)
    S0 = float(prices[asset].iloc[-1])
    K = moneyness * S0

    hist_sigma = float(train_returns[asset].tail(252).std() * math.sqrt(252))
    C0, delta0 = bs_call_price_delta(S0, K, T0, risk_free_rate, hist_sigma)

    S_end = S0 * np.exp(asset_windows.sum(axis=1))
    sigma_scenario = realized_vol(asset_windows)
    C_end, delta_end = bs_call_price_delta(S_end, K, T1, risk_free_rate, sigma_scenario)

    stock_pnl = S_end - S0
    option_pnl = C_end - C0
    combined_pnl = stock_pnl + option_pnl
    initial_value = S0 + C0

    return pd.DataFrame(
        {
            "dataset": name,
            "S_end": S_end,
            "realized_vol": sigma_scenario,
            "call_price_end": C_end,
            "delta_end": delta_end,
            "stock_pnl": stock_pnl,
            "option_pnl": option_pnl,
            "stock_plus_call_pnl": combined_pnl,
            "stock_plus_call_return": combined_pnl / initial_value,
            "initial_stock_price": S0,
            "strike": K,
            "initial_call_price": C0,
            "initial_delta": delta0,
            "historical_sigma_used_at_start": hist_sigma,
        }
    )


def var_cvar_from_pnl_returns(pnl_returns: pd.Series | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    losses = -np.asarray(pnl_returns)
    var = np.quantile(losses, 1 - alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)
