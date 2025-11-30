# ============================
# file: growth_kernel.py
# Core Transformation Kernel: GrowthKernel
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrowthKernel(nn.Module):
    """
    GrowthKernel:
    Trace を受けた瞬間、
    - Self の幾何学的表現 (self_state)
    - Metric (distance / sensitivity structure)
    を同時に変形させる「成長核」。

    本質:
        Self とは位置ではなく、
        『解釈の構造』そのものが変形する空間。
    """

    def __init__(self, dim, eta_metric=0.05, eta_self=0.1, device="cpu"):
        super().__init__()
        self.dim = dim
        self.eta_metric = eta_metric  # Metric の変形率
        self.eta_self = eta_self  # Self の変形率
        self.device = device

    # ==========================================================
    def forward(self, self_state: torch.Tensor,
                metric: torch.Tensor,
                trace: torch.Tensor,
                plasticity: float):
        """
        Self と Metric を同時に変形する核。

        self_state:   (d,)   現在の自己中心
        metric:       (d,d)  心的距離構造
        trace:        (d,)   Traceベクトル
        plasticity:   (0-1)  Traceを取り込める柔軟度（防衛的なら低い）

        Return:
            new_state, new_metric
        """
        trace = trace.to(self.device)
        trace_norm = F.normalize(trace, dim=-1)

        # ===== Self 更新 =====
        new_self = self_state + self.eta_self * plasticity * trace_norm
        new_self = F.normalize(new_self, dim=-1)

        # ===== Metric 更新 (rank-1 update) =====
        outer = torch.outer(trace_norm, trace_norm)
        new_metric = metric + self.eta_metric * plasticity * outer
        new_metric = new_metric + 1e-4 * torch.eye(self.dim, device=self.device)  # 安定化

        return new_self, new_metric
