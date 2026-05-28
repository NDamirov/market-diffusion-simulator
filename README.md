# Diffusion-Based Stock Scenario Generator

This project is a prototype for generating short-term multi-asset stock market scenarios.
It is inspired by the paper **Multi-Asset Spot and Option Market Simulation**.

The repository implements:

- an unconditional DDPM-style diffusion model for 20-day stock return windows;
- a conditional DDPM that can generate `calm`, `normal`, or `stress` volatility regimes;
- a Gaussian baseline;
- distribution, correlation, VaR, and CVaR evaluation;
- a simple Black-Scholes option-pricing layer on top of generated stock paths.

## Repository Layout

```text
Final Project/
  README.md
  requirements.txt
  pyproject.toml
  diffusion_stock_scenario_generator.ipynb
  src/market_diffusion/
    config.py        # project config, seeding, device
    data.py          # yfinance download, synthetic fallback, returns
    windows.py       # rolling windows, scaling, volatility regimes
    diffusion.py     # DDPM models, training, sampling
    evaluation.py    # statistics, correlations, VaR/CVaR, ACF
    options.py       # Black-Scholes layer
    plotting.py      # notebook plotting helpers
```

## How to Run

Open `diffusion_stock_scenario_generator.ipynb` from the repository root and select a Python environment with the dependencies installed:

```bash
pip install -r requirements.txt
```

## Main DL Idea

The base model learns:

```text
epsilon_theta(x_t, t)
```

where `x_t` is a noisy flattened multi-asset return window.

The stronger DL extension is a conditional diffusion model:

```text
epsilon_theta(x_t, t, regime)
```

where `regime` is one of:

```text
calm / normal / stress
```

Regimes are built automatically from realized volatility quantiles. This gives controllable market scenario generation rather than only unconditional sampling.
