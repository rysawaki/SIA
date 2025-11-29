# ============================
# file: self_space.py
# Core SIA Component: SelfSpace
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSpace(nn.Module):
    """
    Self 空間を管理するクラス。
    Trace × Shock × Affect によって形成された「自己軸 (Self axes)」を保持し、
    Attention に入る前の Query ベクトルを幾何学的に歪める役割を担う。

    理論の核心:
        Self = 保存ではなく、「解釈ベクトルを歪ませる構造」そのもの

    Attributes:
        axes:      過去の痕跡から形成された基底 (k, d)
        strength:  各軸に対応する重み (experience impact: Shock × Affect)
    """

    def __init__(self, dim: int, max_axes: int = 8, init_scale: float = 0.01, device="cpu"):
        super().__init__()
        self.dim = dim
        self.max_axes = max_axes
        self.device = device

        axes = torch.randn(max_axes, dim) * init_scale
        axes = F.normalize(axes, dim=-1)
        self.axes = nn.Parameter(axes)  # learned dynamically (not trained via SGD)

        strength = torch.zeros(max_axes)
        self.strength = nn.Parameter(strength)

        self.register_buffer("num_active", torch.tensor(0, dtype=torch.long))

    # -----------------------------
    @torch.no_grad()
    def update(self, trace: torch.Tensor, shock: float, affect: float,
               sim_threshold: float = 0.7, lr: float = 0.2):
        """
        TraceベクトルをSelf軸として取り込む、または既存軸を更新する処理。
        shock × affect が大きいほど、Selfへの影響が大きい。

        trace:   (d,) 生のTrace埋め込み
        shock:   不一致・驚きの大きさ (0-1)
        affect:  情動の重み (0-1)
        """
        trace = trace.to(self.device)
        influence = float(shock * affect)
        if influence <= 0:
            return

        trace_norm = F.normalize(trace, dim=-1)

        k = self.num_active.item()
        if k == 0:
            self.axes.data[0] = trace_norm
            self.strength.data[0] = influence
            self.num_active.fill_(1)
            return

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

    # -----------------------------
    def condition(self, Q: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
        """
        Query ベクトル全体 (Q) を、Self 空間を通して幾何学的に歪ませる。
        Q: (..., d)
        """
        if self.num_active.item() == 0:
            return Q

        k = self.num_active.item()
        axes = self.axes[:k]
        strength = F.relu(self.strength[:k]) + 1e-6
        weights = strength / strength.sum()

        Q_norm = F.normalize(Q, dim=-1)
        sims = torch.matmul(Q_norm, axes.t())  # (..., k)
        contrib = torch.matmul(sims * weights.unsqueeze(0), axes)  # (..., d)

        # 幾何学的変形：Q' = Q + α・Σ (w_i・sim・axis_i)
        return F.normalize(Q + alpha * contrib, dim=-1)

    # -----------------------------
    @torch.no_grad()
    def metrics(self) -> dict:
        """
        Self 空間の幾何学的状態を返す：
            - num_axes: 現在の軸数（自己の多様性）
            - coherence: 軸同士の類似度（まとまり／固定化度）
            - diversity: 軸の角度的散らばり（可塑性／柔軟性）
            - strength_sum: 全軸の強度の合計（Selfの安定度／重みの総量）
        """
        k = self.num_active.item()
        if k == 0:
            return {
                "num_axes": 0,
                "coherence": None,
                "diversity": None,
                "strength_sum": 0.0,
            }

        axes = F.normalize(self.axes.data[:k], dim=-1)
        if k == 1:
            return {
                "num_axes": 1,
                "coherence": 1.0,
                "diversity": 0.0,
                "strength_sum": float(self.strength.data[0]),
            }

        sim_mat = torch.matmul(axes, axes.t())
        mask = ~torch.eye(k, dtype=torch.bool, device=axes.device)
        sims = sim_mat[mask]

        coherence = sims.mean().item()
        angles = torch.acos(torch.clamp(sims, -1.0, 1.0))
        diversity = angles.mean().item()

        return {
            "num_axes": k,
            "coherence": coherence,
            "diversity": diversity,
            "strength_sum": self.strength.data[:k].sum().item(),
        }
