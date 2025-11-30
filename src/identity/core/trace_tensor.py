"""
trace_tensor.py

Core implementation of TraceTensor for SIA (Self-Imprint Attribution).

This module defines a geometric "trace tensor" that:
  - Stores affective / experiential imprints as a low-rank tensor
  - Updates over time via imprint + decay dynamics
  - Deforms Self-space vectors to express how trace warps identity geometry

The design is intentionally minimal so it can be:
  - Plugged into Self-space modules
  - Wrapped by Trace Tensor Attention layers
  - Used in small 2D toy demos and high-D transformer models alike
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TraceTensorConfig:
    d_model: int                  # Dimensionality of Self / latent space
    rank: int = 4                 # Number of trace modes
    init_scale: float = 0.01      # Scale for tensor initialization
    decay: float = 0.999          # Default decay factor for updates
    lr_imprint: float = 0.01      # Default learning rate for imprint
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


class TraceTensor(nn.Module):
    """
    TraceTensor: geometric memory of imprints in Self-space.

    T \in R^{R x d x d}  (R = rank)

    Intuition:
      - Each slice T[r] is a "trace mode" that warps Self-space.
      - Imprints add structured outer-products (Δs ⊗ Δs) into T.
      - Decay slowly forgets past traces but never fully erases them
        (if decay < 1.0).

    Typical usage:
      1. Initialize with d_model (dimension of Self-space)
      2. For each event (e_t, s_t, u_t):
           imprint = trace_tensor.build_imprint(s_t, u_t, strength)
           trace_tensor.update(imprint)
      3. To see how Self is warped:
           s_deformed = trace_tensor.deform_self(s_t)
    """

    def __init__(self, cfg: TraceTensorConfig):
        super().__init__()
        self.cfg = cfg

        factory_kwargs = {}
        if cfg.device is not None:
            factory_kwargs["device"] = cfg.device
        if cfg.dtype is not None:
            factory_kwargs["dtype"] = cfg.dtype

        # T: (rank, d, d)
        T = torch.randn(cfg.rank, cfg.d_model, cfg.d_model, **factory_kwargs)
        T = T * cfg.init_scale
        self.T = nn.Parameter(T)

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def rank(self) -> int:
        return self.cfg.rank

    @torch.no_grad()
    def reset(self) -> None:
        """Re-initialize trace tensor (for experiments)."""
        self.T.data.normal_(mean=0.0, std=self.cfg.init_scale)

    @torch.no_grad()
    def update(
        self,
        imprint: torch.Tensor,
        decay: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """
        In-place update of trace tensor.

        Args:
            imprint: Tensor with the same shape as T (rank, d, d)
            decay:   Exponential decay factor in (0,1]; closer to 1.0 = slower forgetting
            lr:      Learning rate for imprint integration
        """
        if decay is None:
            decay = self.cfg.decay
        if lr is None:
            lr = self.cfg.lr_imprint

        if imprint.shape != self.T.shape:
            raise ValueError(
                f"Imprint shape {imprint.shape} must match trace tensor shape {self.T.shape}"
            )

        # Exponential decay + additive imprint
        self.T.data.mul_(decay).add_(imprint, alpha=lr)

    def build_imprint_from_pair(
        self,
        s_before: torch.Tensor,
        s_after: torch.Tensor,
        strength: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Build an imprint tensor from a change in Self: Δs = s_after - s_before.

        Idea:
          - Shock or meaningful event is represented as a displacement Δs
          - Imprint energy is proportional to outer product Δs ⊗ Δs
          - The same Δs is broadcast to all trace modes (rank)

        Args:
            s_before: (d,) Self vector before event
            s_after:  (d,) Self vector after event
            strength: scalar scale for imprint energy
            normalize: if True, normalize Δs to unit length before outer product

        Returns:
            imprint: (rank, d, d)
        """
        if s_before.ndim != 1 or s_after.ndim != 1:
            raise ValueError("s_before and s_after must be 1D tensors (shape: (d,))")

        delta = s_after - s_before
        if normalize:
            norm = delta.norm(p=2) + 1e-8
            delta = delta / norm

        # Outer product Δs ⊗ Δs -> (d, d)
        outer = torch.ger(delta, delta) * strength  # (d, d)

        # Broadcast to all rank modes
        imprint = outer.unsqueeze(0).expand(self.rank, -1, -1).contiguous()
        return imprint

    def deform_self(self, s: torch.Tensor) -> torch.Tensor:
        """
        Apply trace-induced geometric deformation to Self vector.

        We contract T over one index:
            y_r = T[r] @ s      for each r
          then aggregate over rank:
            y = mean_r y_r

        Args:
            s: (d,) or (batch, d)

        Returns:
            y: same shape as s
        """
        if s.ndim == 1:
            # (rank, d, d) x (d,) -> (rank, d)
            y_r = torch.einsum("r i j, j -> r i", self.T, s)
            y = y_r.mean(dim=0)
            return y

        elif s.ndim == 2:
            # (batch, d)
            # We treat each batch element independently
            # (b, d) -> (b, rank, d)
            s_exp = s.unsqueeze(1)  # (b, 1, d)
            # (rank, d, d) -> (1, rank, d, d)
            T_exp = self.T.unsqueeze(0)  # (1, r, d, d)

            # y[b, r, i] = sum_j T[r, i, j] * s[b, j]
            y_br = torch.einsum("b r i j, b j -> b r i", T_exp, s_exp.squeeze(1))
            y = y_br.mean(dim=1)  # (b, d)
            return y

        else:
            raise ValueError("s must have shape (d,) or (batch, d)")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Alias for deform_self so it can be used as a nn.Module in pipelines.

        Example:
            trace = TraceTensor(cfg)
            s_deformed = trace(s)
        """
        return self.deform_self(s)


# --- Simple 2D demo utilities (for experiments / notebooks) ---

def demo_build_trace_and_deform_2d(
    steps: int = 5,
    strength: float = 1.0,
    seed: int = 42,
):
    """
    Minimal 2D toy example:
      - Self-space is 2D
      - We generate a line of Self states and treat them as successive updates
      - After accumulating trace, we deform a grid to visualize the warp

    This function is intended for use in notebooks / scripts, not as a training loop.
    """
    import math
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)

    cfg = TraceTensorConfig(d_model=2, rank=4, init_scale=0.0)
    trace = TraceTensor(cfg)

    # 1. Generate a simple trajectory in Self-space
    s_list = []
    for t in range(steps + 1):
        angle = 2 * math.pi * t / steps
        s = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        s_list.append(s)

    # 2. Accumulate imprints along the trajectory
    for t in range(steps):
        s_before = s_list[t]
        s_after = s_list[t + 1]
        imprint = trace.build_imprint_from_pair(
            s_before, s_after, strength=strength, normalize=True
        )
        trace.update(imprint)

    # 3. Visualize how the trace deforms a grid
    grid_lin = torch.linspace(-1.5, 1.5, steps * 4)
    xs, ys = torch.meshgrid(grid_lin, grid_lin, indexing="xy")
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (N, 2)

    with torch.no_grad():
        deformed = trace(points)  # (N, 2)

    x_orig = points[:, 0].reshape(xs.shape)
    y_orig = points[:, 1].reshape(ys.shape)
    x_def = deformed[:, 0].reshape(xs.shape)
    y_def = deformed[:, 1].reshape(ys.shape)

    plt.figure(figsize=(6, 6))
    # Plot original grid as light lines
    for i in range(xs.shape[0]):
        plt.plot(x_orig[i].numpy(), y_orig[i].numpy(), linewidth=0.5, alpha=0.3)
        plt.plot(x_orig[:, i].numpy(), y_orig[:, i].numpy(), linewidth=0.5, alpha=0.3)

    # Plot deformed grid
    for i in range(xs.shape[0]):
        plt.plot(x_def[i].numpy(), y_def[i].numpy(), linewidth=1.0)
        plt.plot(x_def[:, i].numpy(), y_def[:, i].numpy(), linewidth=1.0)

    plt.scatter(
        [s[0].item() for s in s_list],
        [s[1].item() for s in s_list],
        s=20,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("TraceTensor 2D Grid Deformation Demo")
    plt.xlabel("Self dim 1")
    plt.ylabel("Self dim 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    return trace


if __name__ == "__main__":
    # Simple smoke test
    cfg = TraceTensorConfig(d_model=4)
    trace = TraceTensor(cfg)

    s_before = torch.randn(4)
    s_after = s_before + 0.5 * torch.randn(4)
    imprint = trace.build_imprint_from_pair(s_before, s_after, strength=1.0)
    trace.update(imprint)

    s = torch.randn(4)
    y = trace(s)
    print("s:", s)
    print("y (deformed):", y)
