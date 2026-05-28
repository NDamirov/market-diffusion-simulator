# Diffusion-Based Stock Scenario Generator

This project is a compact deep learning prototype for generating short-term multi-asset stock market scenarios.
It is inspired by the paper **Multi-Asset Spot and Option Market Simulation**, but deliberately keeps the implementation feasible for a course project.

The repository implements:

- an unconditional DDPM-style diffusion model for 20-day stock return windows;
- a conditional DDPM that can generate `calm`, `normal`, or `stress` volatility regimes;
- a Gaussian baseline;
- distribution, correlation, VaR, and CVaR evaluation;
- a simple Black-Scholes option-pricing layer on top of generated stock paths.

Important limitation: this is not a full option market simulator. It does not learn option chains, implied volatility surfaces, or arbitrage-free constraints.

## Repository Layout

```text
Final Project/
  README.md
  requirements.txt
  pyproject.toml
  src/market_diffusion/
    config.py        # project config, seeding, device
    data.py          # yfinance download, synthetic fallback, returns
    windows.py       # rolling windows, scaling, volatility regimes
    diffusion.py     # DDPM models, training, sampling
    evaluation.py    # statistics, correlations, VaR/CVaR, ACF
    options.py       # Black-Scholes layer
    plotting.py      # notebook plotting helpers
  notebooks/
    final_project_colab.ipynb
    legacy_single_notebook.ipynb
  scripts/
    run_smoke.py
```

## Colab Usage

Clone or upload the full repository, then open:

```text
notebooks/final_project_colab.ipynb
```

The notebook adds `src/` to `sys.path`, so the code is imported from modules instead of being defined inline.

## Local Smoke Test

From the repository root:

```bash
python scripts/run_smoke.py
```

The smoke test uses synthetic data and tiny training settings. It verifies that the modules import, train, and sample without running the full experiment.

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
