# ============================
# file: visualize_selfspace.py
# 可視化：Self-conditioning による Query 歪みと Self軸の構造
# ============================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def project_to_2d(vectors: torch.Tensor):
    """
    ベクトル群 (N, d) を PCA によって 2次元へ射影する。
    戻り値は numpy 配列 shape: (N, 2)
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors.detach().cpu().numpy())


def visualize_query_deformation(Q, Q_cond, axes=None, save_path=None):
    """
    Query と Self-conditioned Query の幾何学的変形を可視化。

    Parameters:
        Q:       (d,) or (N, d)   元のクエリ（単一 or 複数）
        Q_cond:  (d,) or (N, d)   Self を通した後のクエリ
        axes:    (k, d) or None   Self軸 (optional)
    """
    if Q.dim() == 1:  # (d,) → (1, d)
        Q = Q.unsqueeze(0)
        Q_cond = Q_cond.unsqueeze(0)

    data = [Q, Q_cond]
    labels = ["Original Q", "Self-conditioned Q"]
    colors = ["blue", "red"]

    if axes is not None:
        data.append(axes)
        labels.append("Self axes")
        colors.append("green")

    data_cat = torch.cat(data, dim=0)  # (N_total, d)
    proj_2d = project_to_2d(data_cat)  # (N_total, 2)

    idx_Q = range(len(Q))
    idx_Qc = range(len(Q), len(Q) + len(Q_cond))
    idx_axes = range(len(Q) + len(Q_cond), proj_2d.shape[0])

    plt.figure(figsize=(7, 6))

    plt.scatter(proj_2d[idx_Q, 0], proj_2d[idx_Q, 1], label="Original Q", s=60, marker='o')
    plt.scatter(proj_2d[idx_Qc, 0], proj_2d[idx_Qc, 1], label="Self-conditioned Q", s=60, marker='x')

    # 変形ベクトルを矢印で表示
    for i in range(len(Q)):
        plt.arrow(
            proj_2d[idx_Q][i, 0], proj_2d[idx_Q][i, 1],
            proj_2d[idx_Qc][i, 0] - proj_2d[idx_Q][i, 0],
            proj_2d[idx_Qc][i, 1] - proj_2d[idx_Q][i, 1],
            color="gray", alpha=0.6, width=0.003
        )

    # Self軸があるなら表示
    if axes is not None:
        plt.scatter(proj_2d[idx_axes, 0], proj_2d[idx_axes, 1], label="Self axes", s=80, marker='^')

    plt.title("Geometric Deformation of Query by Self")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_self_axes(axes, save_path=None):
    """
    SelfSpace 内の軸（多様性／方向性）をプロット。
    Self が固定化に向かっているのか、広がっているのか可視化。
    """
    if axes is None or len(axes) == 0:
        print("No Self axes to visualize.")
        return

    proj_2d = project_to_2d(axes)  # (k, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=80, c='green')

    for i, (x, y) in enumerate(proj_2d):
        plt.text(x, y, f"Axis {i}", fontsize=9)

    plt.title("Distribution of Self Axes (PCA Projection)")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def compare_before_after(Q_before, Q_after, title="Before vs After Self Update"):
    """
    Self 更新前後で Query がどの程度変わったかを可視化。
    （Selfの学習効果を検証できる）
    """
    data = torch.cat([Q_before, Q_after], dim=0)
    proj_2d = project_to_2d(data)

    N = len(Q_before)
    plt.figure(figsize=(7, 6))
    plt.scatter(proj_2d[:N, 0], proj_2d[:N, 1], label="Before Update", s=60, marker='o')
    plt.scatter(proj_2d[N:, 0], proj_2d[N:, 1], label="After Update", s=60, marker='x')

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
