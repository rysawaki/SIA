import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. SelfSpace クラス（簡易版）
# =========================

class SelfSpace(nn.Module):
    def __init__(self, dim: int, max_axes: int = 6, init_scale: float = 0.01, device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.max_axes = max_axes
        self.device = device

        axes = torch.randn(max_axes, dim) * init_scale
        axes = F.normalize(axes, dim=-1)
        self.axes = nn.Parameter(axes)

        strength = torch.zeros(max_axes)
        self.strength = nn.Parameter(strength)

        self.register_buffer("num_active", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _ensure_initialized(self):
        if self.num_active.item() == 0:
            self.num_active.fill_(1)
            self.axes.data[0] = F.normalize(self.axes.data[0], dim=-1)
            self.strength.data[0] = 1.0

    @torch.no_grad()
    def update(self, trace: torch.Tensor, shock: float, affect: float,
               sim_threshold: float = 0.7, lr: float = 0.2):
        self._ensure_initialized()

        trace = trace.to(self.device)
        trace_norm = F.normalize(trace, dim=-1)

        k = self.num_active.item()
        active_axes = self.axes.data[:k]  # (k, d)

        sims = F.cosine_similarity(active_axes, trace_norm.unsqueeze(0), dim=-1)  # (k,)
        max_sim, idx = sims.max(dim=0)

        influence = float(shock * affect)
        if influence <= 0:
            return

        if max_sim > sim_threshold:
            # 既存軸を更新
            i = idx.item()
            old_axis = active_axes[i]
            new_axis = F.normalize(
                (1 - lr * influence) * old_axis + (lr * influence) * trace_norm,
                dim=-1
            )
            self.axes.data[i] = new_axis
            self.strength.data[i] += influence
        else:
            # 新しい軸として追加
            if k < self.max_axes:
                self.axes.data[k] = trace_norm
                self.strength.data[k] = influence
                self.num_active.add_(1)
            else:
                weakest_idx = torch.argmin(self.strength.data[:k]).item()
                self.axes.data[weakest_idx] = trace_norm
                self.strength.data[weakest_idx] = influence

    def condition(self, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        query ベクトル x を Self によって歪ませる:
          x' = x + α * Σ_i w_i * sim(x, axis_i) * axis_i
        """
        self._ensure_initialized()

        k = self.num_active.item()
        axes = self.axes[:k]  # (k, d)
        strength = F.relu(self.strength[:k]) + 1e-6  # (k,)
        weights = strength / strength.sum()

        x_norm = F.normalize(x, dim=-1)  # (..., d)
        sims = torch.matmul(x_norm, axes.t())  # (..., k)

        contrib = torch.matmul(sims * weights.unsqueeze(0), axes)  # (..., d)
        return x + alpha * contrib

    @torch.no_grad()
    def metrics(self):
        self._ensure_initialized()

        k = self.num_active.item()
        axes = F.normalize(self.axes.data[:k], dim=-1)

        if k == 1:
            diversity = 0.0
            coherence = 1.0
        else:
            sim_mat = torch.matmul(axes, axes.t())
            mask = ~torch.eye(k, dtype=torch.bool, device=axes.device)
            sims = sim_mat[mask]
            coherence = sims.mean().item()
            angles = torch.acos(torch.clamp(sims, -1.0, 1.0))
            diversity = angles.mean().item()

        strength_sum = self.strength.data[:k].sum().item()
        return {
            "num_axes": k,
            "coherence": coherence,
            "diversity": diversity,
            "strength_sum": strength_sum
        }


# =========================
# 2. シンプルな Attention 実装
# =========================

def simple_attention(query: torch.Tensor, keys: torch.Tensor):
    """
    query: (d,)
    keys:  (n, d)
    戻り値: attn_weights: (n,), attn_output: (d,)
    """
    q = F.normalize(query, dim=-1)
    k = F.normalize(keys, dim=-1)  # コサイン類似度ベースにする

    scores = torch.matmul(k, q)  # (n,)
    attn_weights = F.softmax(scores, dim=-1)  # (n,)
    attn_output = torch.matmul(attn_weights, keys)  # (d,)

    return attn_weights, attn_output


# =========================
# 3. デモ: Self あり / なしで Attention を可視化
# =========================

def run_self_conditioned_attention_demo(
    dim: int = 16,
    num_tokens: int = 10,
    num_experiences: int = 30,
    seed: int = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 「文」の各トークンの埋め込み（疑似）
    keys = torch.randn(num_tokens, dim)

    # 視点となる query（例: ある質問の内部表現）
    query = torch.randn(dim)

    # Self 空間
    self_space = SelfSpace(dim=dim, max_axes=6, device="cpu")

    # --- 経験前の Attention ---
    attn_before, _ = simple_attention(query, keys)

    # Selfにいくつかの Trace / Shock / Affect 経験を流し込む
    for t in range(num_experiences):
        trace = torch.randn(dim)  # ここではランダムだが、本来は state から生成される埋め込み
        shock = float(torch.rand(()))           # 0〜1
        affect = float(torch.rand(()))          # 0〜1
        self_space.update(trace, shock=shock, affect=affect)

    print("Self metrics after experiences:", self_space.metrics())

    # Selfで query を歪ませる
    query_cond = self_space.condition(query.unsqueeze(0)).squeeze(0)

    # --- Self-conditioned Attention ---
    attn_after, _ = simple_attention(query_cond, keys)

    # =========================
    # 4. 可視化
    # =========================

    indices = np.arange(num_tokens)

    plt.figure(figsize=(10, 4))

    # (1) 棒グラフで Before / After 比較
    plt.subplot(1, 2, 1)
    width = 0.35
    plt.bar(indices - width/2, attn_before.detach().numpy(), width=width, label="Before Self")
    plt.bar(indices + width/2, attn_after.detach().numpy(), width=width, label="After Self")
    plt.xlabel("Token index")
    plt.ylabel("Attention weight")
    plt.title("Attention weights per token")
    plt.legend()
    plt.tight_layout()

    # (2) Heatmap 風に Before / After を並べて表示
    plt.subplot(1, 2, 2)
    attn_matrix = torch.stack([attn_before, attn_after], dim=0).detach().numpy()  # (2, n)
    plt.imshow(attn_matrix, aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.yticks([0, 1], ["Before", "After"])
    plt.xticks(indices, [str(i) for i in indices])
    plt.title("Attention heatmap (Before / After Self)")
    plt.tight_layout()

    plt.suptitle("Self-conditioned Attention Demo", y=1.05, fontsize=14)
    plt.show()


if __name__ == "__main__":
    run_self_conditioned_attention_demo()
