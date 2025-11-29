#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trace Tensor Attention 可視化デモ

・2次元のトークン埋め込み Q, K を生成
・通常の Attention 行列を計算
・Trace Tensor T で Q, K を幾何学的に変形
・変形前後の Attention 行列と、2D空間の歪みを可視化

依存:
    pip install torch matplotlib numpy
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_tokens_on_circle(num_tokens: int = 8, radius: float = 1.0, seed: int = 42):
    """
    円周上にトークンを配置して 2D ベクトルをつくる。
    視覚的にわかりやすいよう、意味空間を円として扱う。

    Returns:
        Q, K: (num_tokens, 2) の torch.Tensor
        labels: list[str] 各トークンのラベル
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    angles = torch.linspace(0, 2 * math.pi, steps=num_tokens + 1)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    pts = torch.stack([x, y], dim=-1)  # (N, 2)

    # このデモでは Q=K として扱う
    Q = pts.clone()
    K = pts.clone()

    labels = [f"T{i}" for i in range(num_tokens)]
    return Q, K, labels


def make_trace_tensor(shock_strength: float = 1.0, rotation_deg: float = 25.0, anisotropy: float = 0.6):
    """
    Trace Tensor T を構成する。
    ・回転
    ・軸方向の伸縮（anisotropy）
    ・ショック強度で全体スケールを調整

    2x2 の線形変換として定義:
        T = R(θ) @ A @ R(-θ) * (1 + shock_strength * α)

    Args:
        shock_strength: 痕跡（Shock）の強さ。大きいほど歪みが強くなる。
        rotation_deg: 歪み軸を回転させる角度 (度数法)。
        anisotropy: x/y 軸の伸縮差。1.0 で等方、0.6 などで長軸・短軸が生まれる。

    Returns:
        T: torch.Tensor, shape (2, 2)
    """
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s],
                      [s,  c]], dtype=torch.float32)

    # x方向に強く伸ばし、yを抑えるなど
    A = torch.tensor([[1.0 + anisotropy, 0.0],
                      [0.0, 1.0 - anisotropy]], dtype=torch.float32)

    T = R @ A @ torch.inverse(R)

    # ショック強度で全体をスケール
    T = torch.eye(2) + shock_strength * (T - torch.eye(2))
    return T


def compute_attention(Q: torch.Tensor, K: torch.Tensor):
    """
    標準的な Scaled Dot-Product Attention のスコア行列のみを計算。
    ソフトマックス前・後の両方を返す。

    Args:
        Q, K: shape (N, d)

    Returns:
        logits: (N, N)
        attn:   (N, N) 行方向 i が「クエリ i の重み分布」
    """
    d_k = Q.size(-1)
    logits = Q @ K.t() / math.sqrt(d_k)  # (N, N)
    attn = F.softmax(logits, dim=-1)
    return logits, attn


def plot_embeddings(ax, Q, Q_prime, labels, title="Embedding Space"):
    """
    2D 埋め込みの変形前後を同じ図上に描画。
    """
    Q = Q.detach().cpu().numpy()
    Qp = Q_prime.detach().cpu().numpy()

    # 元の位置
    ax.scatter(Q[:, 0], Q[:, 1], alpha=0.7, label="Before (Q)")
    for i, lab in enumerate(labels):
        ax.text(Q[i, 0] + 0.02, Q[i, 1] + 0.02, lab, fontsize=9)

    # 変形後の位置
    ax.scatter(Qp[:, 0], Qp[:, 1], marker="x", alpha=0.9, label="After (Q')")
    for i, lab in enumerate(labels):
        ax.text(Qp[i, 0] + 0.02, Qp[i, 1] - 0.04, lab + "'", fontsize=9, color="red")

    # 変形の矢印
    for i in range(Q.shape[0]):
        ax.arrow(Q[i, 0], Q[i, 1],
                 Qp[i, 0] - Q[i, 0],
                 Qp[i, 1] - Q[i, 1],
                 head_width=0.03, alpha=0.3, length_includes_head=True)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.legend(loc="upper right")


def plot_attention_matrix(ax, attn, title="Attention"):
    """
    Attention 行列をヒートマップとして描画。
    """
    attn_np = attn.detach().cpu().numpy()
    im = ax.imshow(attn_np, interpolation="nearest", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Key index j")
    ax.set_ylabel("Query index i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run_demo(num_tokens: int = 8,
             shock_strength: float = 1.0,
             rotation_deg: float = 30.0,
             anisotropy: float = 0.6,
             save_path: str = None):
    """
    Trace Tensor Attention の挙動を可視化するデモ。

    ・num_tokens: トークン数（小さいほど見やすい）
    ・shock_strength: 歪みの強さ
    ・rotation_deg: 歪み軸の回転
    ・anisotropy: 伸縮の非対称性
    """
    # 1. 2Dトークン生成
    Q, K, labels = make_tokens_on_circle(num_tokens=num_tokens, radius=1.0)

    # 2. 変形前の Attention
    _, attn_before = compute_attention(Q, K)

    # 3. Trace Tensor を構成
    T = make_trace_tensor(
        shock_strength=shock_strength,
        rotation_deg=rotation_deg,
        anisotropy=anisotropy,
    )

    # 4. Q, K を幾何学的に変形
    #    Q': (N, 2) = Q @ T^T
    Q_prime = Q @ T.t()
    K_prime = K @ T.t()
    _, attn_after = compute_attention(Q_prime, K_prime)

    # 5. 描画
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    plot_embeddings(
        axes[0],
        Q,
        Q_prime,
        labels,
        title=f"Trace Tensor Deformation\nshock={shock_strength}, rot={rotation_deg}°, aniso={anisotropy}",
    )

    plot_attention_matrix(axes[1], attn_before, title="Attention Before (standard)")
    plot_attention_matrix(axes[2], attn_after, title="Attention After (Trace Tensor)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # パラメータをいじると「意味空間がどう歪み、注意がどう変わるか」が見える
    run_demo(
        num_tokens=8,
        shock_strength=1.5,   # 痕跡（Shock）の強さ
        rotation_deg=30.0,    # 歪み方向
        anisotropy=0.7,       # どれだけ一方向に引き伸ばすか
        save_path=None        # "trace_tensor_attention_demo.png" にすれば保存
    )
