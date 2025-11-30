# ============================
# file: self_space_v2.py
# Core SIA Component: SelfSpace with Metric Update
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSpace(nn.Module):
    """
    Self-Space v2:
    Trace によって変形する『幾何学的な自己空間』を実装。
    v1 では軸 (axes) の蓄積のみだったが、
    v2 では Metric (何が近く何が遠いか) を Trace から学習する。

    Key Concept:
        Self とは、記憶の格納庫ではなく、
        経験により歪み続ける『距離の構造（Metric）』である。
    """

    def __init__(self, dim: int, max_axes: int = 8, init_scale: float = 0.01, device="cpu"):
        super().__init__()
        self.dim = dim
        self.max_axes = max_axes
        self.device = device

        # Self axes（v1と同じ）
        axes = torch.randn(max_axes, dim) * init_scale
        axes = F.normalize(axes, dim=-1)
        self.axes = nn.Parameter(axes)

        self.strength = nn.Parameter(torch.zeros(max_axes))
        self.register_buffer("num_active", torch.tensor(0, dtype=torch.long))

        # 🆕 Metric（初期状態は単位行列＝等方的な距離）
        self.metric = nn.Parameter(torch.eye(dim, device=device))

    # ==========================================================
    @torch.no_grad()
    def update(self, trace: torch.Tensor, shock: float, affect: float,
               sim_threshold: float = 0.7, lr: float = 0.2, eta: float = 0.05):
        """
        Trace を Self に取り込む処理（v1と同じ）＋ Metric 更新（v2拡張）

        shock × affect が大きいほど、Self構造への影響は大きい。
        """
        trace = trace.to(self.device)
        influence = float(shock * affect)
        if influence <= 0:
            return

        trace_norm = F.normalize(trace, dim=-1)

        # === 1) まず Self軸更新 (v1)
        k = self.num_active.item()
        if k == 0:
            self.axes.data[0] = trace_norm
            self.strength.data[0] = influence
            self.num_active.fill_(1)
        else:
            active_axes = self.axes.data[:k]
            sims = F.cosine_similarity(active_axes, trace_norm.unsqueeze(0), dim=-1)
            max_sim, idx = sims.max(dim=0)

            if max_sim > sim_threshold:
                i = idx.item()
                old_axis = active_axes[i]
                new_axis = F.normalize(
                    (1 - lr * influence) * old_axis + (lr * influence) * trace_norm,
                    dim=-1
                )
                self.axes.data[i] = new_axis
                self.strength.data[i] += influence
            else:
                if k < self.max_axes:
                    self.axes.data[k] = trace_norm
                    self.strength.data[k] = influence
                    self.num_active.add_(1)
                else:
                    weakest_idx = torch.argmin(self.strength.data[:k]).item()
                    self.axes.data[weakest_idx] = trace_norm
                    self.strength.data[weakest_idx] = influence

        # === 2) Metric の更新（v2の核）
        self.update_metric(trace_norm, influence, eta)

    # ==========================================================
    @torch.no_grad()
    def update_metric(self, trace_norm: torch.Tensor, influence: float, eta: float):
        """
        Metric（心理的距離構造）を Trace に応じて変形させる。

        理論:
            g_{t+1} = g_t + η * influence * (trace ⊗ trace)

        trace ⊗ trace = rank-1 update → 自己が経験の方向に感度を持つようになる
        """
        outer = torch.ger(trace_norm, trace_norm)
        self.metric.data = self.metric.data + eta * influence * outer

        # 安定性確保（正定値性の維持）
        self.metric.data = self.metric.data + 1e-4 * torch.eye(self.dim, device=self.device)

    # ==========================================================
    def condition(self, Q: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Query を Self-Space Metric で幾何学的に歪ませる。
        """
        if self.num_active.item() == 0:
            return Q

        Q_proj = torch.matmul(Q, self.metric)  # Metricによる幾何変形
        return F.normalize((1 - alpha) * Q + alpha * Q_proj, dim=-1)

    # ==========================================================
    @torch.no_grad()
    def metrics(self) -> dict:
        """
        Self構造の幾何学的状態を返す（v2 → Metricの情報を含む）
        """
        k = self.num_active.item()
        return {
            "num_axes": k,
            "strength_sum": float(self.strength.data[:k].sum()) if k > 0 else 0.0,
            "metric_trace": torch.trace(self.metric).item(),
            "metric_norm": torch.norm(self.metric).item(),
        }
