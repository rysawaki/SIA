#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Self-Conditioned Trace Tensor Attention Visualization (Animated)

🔹 Self-Conditioned Attention (SCA):
    Attentionスコアに自己状態(Self)が直接影響する:  e_ij += uᵀ s_t

🔹 Trace Tensor Attention (TTA):
    過去の経験(Shock)が空間そのものを変形させる: Q' = TQ, K' = TK

🔹 本コードの可視化内容:
    - Step進行とともにTrace Tensorが空間を歪める
    - Selfによって特定トークンのAttentionが強調される
    - Identity-like biasが時間とともに進化する様子をGIF化

依存:
    pip install torch matplotlib numpy imageio
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio


# -----------------------------
# Token配置 (2D円周)
# -----------------------------
def make_tokens(num_tokens=8, radius=1.0):
    angles = torch.linspace(0, 2 * math.pi, steps=num_tokens + 1)[:-1]
    pts = torch.stack([radius * torch.cos(angles),
                       radius * torch.sin(angles)], dim=-1)
    labels = [f"T{i}" for i in range(num_tokens)]
    return pts.clone(), pts.clone(), labels


# -----------------------------
# Attentionスコア計算 (SCA対応)
# -----------------------------
def compute_attention(Q, K, self_vec=None, u=None):
    d_k = Q.size(-1)
    logits = Q @ K.t() / math.sqrt(d_k)

    if self_vec is not None and u is not None:
        bias = torch.matmul(u, self_vec)  # uᵀ s_t（スカラー）
        logits += bias

    return F.softmax(logits, dim=-1)


# -----------------------------
# Trace Tensor の更新
# -----------------------------
def update_trace_tensor(T, shock_center, shock_strength=0.3, decay=0.98):
    dist = torch.norm(T - shock_center).item()
    T = T + shock_strength * torch.tensor([[dist, -0.5 * dist],
                                           [0.5 * dist, -dist]])
    return decay * T


# -----------------------------
# 可視化フレーム生成
# -----------------------------
def plot_frame(Q, Qp, attn, step, filename, self_vec):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(Q[:, 0], Q[:, 1], label="Before", alpha=0.6)
    axes[0].scatter(Qp[:, 0], Qp[:, 1], label="After", marker='x', alpha=0.9, color='red')
    for i in range(Q.size(0)):
        axes[0].arrow(Q[i,0], Q[i,1], Qp[i,0]-Q[i,0], Qp[i,1]-Q[i,1],
                      alpha=0.3, head_width=0.03)

    axes[0].set_title(f"Trace Tensor Deformation\n(step={step})\nSelf={self_vec.tolist()}")
    axes[0].set_aspect("equal")

    im = axes[1].imshow(attn.detach().numpy(), origin='lower')
    axes[1].set_title("Attention After (SCA + Trace Tensor)")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


# -----------------------------
# メイン実行ループ
# -----------------------------
def run():
    Q, K, labels = make_tokens(num_tokens=8)
    T = torch.eye(2)

    u = torch.tensor([0.8, 0.5])  # Self の投影方向ベクトル
    self_vec = torch.tensor([0.3, 0.7])  # 初期 Self 状態

    frames = []
    for step in range(15):

        shock_center = torch.tensor([0.4, -0.2])
        T = update_trace_tensor(T, shock_center,
                                shock_strength=0.35,
                                decay=0.96)

        Qp = Q @ T.t()
        Kp = K @ T.t()

        attn = compute_attention(Qp, Kp, self_vec=self_vec, u=u)

        fname = f"sca_tta_{step:03d}.png"
        plot_frame(Q, Qp, attn, step, fname, self_vec)
        frames.append(imageio.imread(fname))

        self_vec = 0.95 * self_vec + torch.tensor([0.02, -0.01])

    imageio.mimsave("../../outputs/12/Self_Trace_Attention.gif", frames, fps=2)
    print("Saved: Self_Trace_Attention.gif")


if __name__ == "__main__":
    run()
