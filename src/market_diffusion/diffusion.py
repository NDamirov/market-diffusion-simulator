from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DenoiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_emb(t)], dim=1))


class ConditionalDenoiseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        n_regimes: int = 3,
        regime_emb_dim: int = 32,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.regime_emb = nn.Embedding(n_regimes + 1, regime_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim + regime_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, regime_label: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x, self.time_emb(t), self.regime_emb(regime_label)], dim=1)
        return self.net(features)


class DiffusionSchedule:
    def __init__(self, n_steps: int, device: torch.device):
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)


def train_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    train_windows: np.ndarray,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    epochs: int,
    log_prefix: str = "Epoch",
) -> list[float]:
    x = torch.tensor(train_windows.reshape(len(train_windows), -1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, schedule.n_steps, (batch.size(0),), device=device)
            noise = torch.randn_like(batch)
            x_t = (
                schedule.sqrt_alpha_bars[t].unsqueeze(1) * batch
                + schedule.sqrt_one_minus_alpha_bars[t].unsqueeze(1) * noise
            )
            pred_noise = model(x_t, t)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            print(f"{log_prefix} {epoch:4d}/{epochs} | loss={mean_loss:.5f}")
    return history


def train_conditional_ddpm(
    model: ConditionalDenoiseMLP,
    schedule: DiffusionSchedule,
    train_windows: np.ndarray,
    regime_labels: np.ndarray,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    epochs: int,
    unknown_regime_label: int,
    label_drop_prob: float = 0.15,
) -> list[float]:
    x = torch.tensor(train_windows.reshape(len(train_windows), -1), dtype=torch.float32)
    y = torch.tensor(regime_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            labels_for_model = labels.clone()
            drop_mask = torch.rand(labels.shape, device=device) < label_drop_prob
            labels_for_model[drop_mask] = unknown_regime_label

            t = torch.randint(0, schedule.n_steps, (batch.size(0),), device=device)
            noise = torch.randn_like(batch)
            x_t = (
                schedule.sqrt_alpha_bars[t].unsqueeze(1) * batch
                + schedule.sqrt_one_minus_alpha_bars[t].unsqueeze(1) * noise
            )
            pred_noise = model(x_t, t, labels_for_model)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            print(f"Conditional epoch {epoch:4d}/{epochs} | loss={mean_loss:.5f}")
    return history


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int,
    input_dim: int,
    window_size: int,
    n_assets: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x = torch.randn(n_samples, input_dim, device=device)
    for step in reversed(range(schedule.n_steps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        x = _reverse_step(x, pred_noise, schedule, step)
    return x.detach().cpu().numpy().reshape(n_samples, window_size, n_assets)


@torch.no_grad()
def sample_conditional_ddpm(
    model: ConditionalDenoiseMLP,
    schedule: DiffusionSchedule,
    n_samples: int,
    input_dim: int,
    window_size: int,
    n_assets: int,
    device: torch.device,
    regime_label: int,
    unknown_regime_label: int,
    guidance_scale: float = 1.5,
) -> np.ndarray:
    model.eval()
    x = torch.randn(n_samples, input_dim, device=device)
    cond_labels = torch.full((n_samples,), regime_label, device=device, dtype=torch.long)
    uncond_labels = torch.full((n_samples,), unknown_regime_label, device=device, dtype=torch.long)

    for step in reversed(range(schedule.n_steps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        pred_cond = model(x, t, cond_labels)
        pred_uncond = model(x, t, uncond_labels)
        pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        x = _reverse_step(x, pred_noise, schedule, step)
    return x.detach().cpu().numpy().reshape(n_samples, window_size, n_assets)


def _reverse_step(
    x: torch.Tensor,
    pred_noise: torch.Tensor,
    schedule: DiffusionSchedule,
    step: int,
) -> torch.Tensor:
    beta_t = schedule.betas[step]
    alpha_t = schedule.alphas[step]
    alpha_bar_t = schedule.alpha_bars[step]
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
    )
    if step > 0:
        return mean + torch.sqrt(beta_t) * torch.randn_like(x)
    return mean
