# sia_boundary_3d.py
# 1D Mini-SIA のロジスティック回帰から得た
# 崩壊境界面 f(α,β,γ)=0 を 3D 曲面として描画するだけのスクリプト

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D)

# ============================================================
# 1. ロジスティック回帰の係数（v14 の出力をそのまま使用）
# ============================================================

# 順番:
# [1, alpha, beta, gamma, alpha^2, alpha*beta,
#  alpha*gamma, beta^2, beta*gamma, gamma^2]
coef = np.array([
    -0.0001,   # 1
    -0.1062,   # alpha
    -0.1062,   # beta
    2.1340,    # gamma
    -0.1085,   # alpha^2
    -0.0815,   # alpha*beta
    1.3475,    # alpha*gamma
    -0.1085,   # beta^2
    1.3475,    # beta*gamma
    1.2400     # gamma^2
], dtype=np.float64)

intercept = -0.7421

# 1項と intercept をまとめて定数項にしてしまう
const_term = coef[0] + intercept

# ============================================================
# 2. 境界式 f(α,β,γ)=0 を γ について解く
# ============================================================
# f(α,β,γ) =
#   1.2400 γ^2
# + (2.1340 + 1.3475 α + 1.3475 β) γ
# - 0.1085 α^2 - 0.0815 αβ - 0.1062 α
# - 0.1085 β^2 - 0.1062 β
# + const_term
#
# → a2 γ^2 + a1(α,β) γ + a0(α,β) = 0 の形

a2 = coef[9]  # gamma^2 = 1.2400

def gamma_crit(alpha, beta):
    """
    崩壊境界 f(α,β,γ)=0 を満たす γ のうち、
    0 以上の実数解（物理的に意味のある方）を返す。
    解が存在しない・範囲外なら np.nan。
    alpha, beta は numpy の配列でもスカラーでもOK（ブロードキャスト対応）。
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    a1 = coef[3] + coef[6] * alpha + coef[8] * beta  # γ の係数
    a0 = (
        coef[4] * alpha**2 +
        coef[5] * alpha * beta +
        coef[1] * alpha +
        coef[7] * beta**2 +
        coef[2] * beta +
        const_term
    )

    disc = a1**2 - 4 * a2 * a0  # 判別式

    # 判別式が負のところは解なし
    disc_clipped = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc_clipped)

    # 2つの解
    gamma1 = (-a1 + sqrt_disc) / (2 * a2)
    gamma2 = (-a1 - sqrt_disc) / (2 * a2)

    # 正で [0, 2] くらいの範囲に入る方を採用する
    gamma = np.where(
        (gamma1 >= 0.0) & (gamma1 <= 2.0),
        gamma1,
        np.where(
            (gamma2 >= 0.0) & (gamma2 <= 2.0),
            gamma2,
            np.nan
        )
    )
    return gamma

# ============================================================
# 3. グリッド上で γ_crit(α,β) を計算し 3D 曲面を描く
# ============================================================

def plot_3d_boundary():
    # α, β の範囲（Mini-SIA で使っていた 0〜1.5 に合わせる）
    alpha_vals = np.linspace(0.0, 1.5, 60)
    beta_vals  = np.linspace(0.0, 1.5, 60)

    A, B = np.meshgrid(alpha_vals, beta_vals)
    G = gamma_crit(A, B)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 有効な点だけを描画（NaN を避ける）
    mask = ~np.isnan(G)
    A_valid = A[mask]
    B_valid = B[mask]
    G_valid = G[mask]

    # 曲面として描くために再グリッド化してもいいが、
    # とりあえず散布＋トライサーフで滑らかな面にする
    surf = ax.plot_trisurf(
        A_valid.ravel(),
        B_valid.ravel(),
        G_valid.ravel(),
        linewidth=0.2,
        antialiased=True,
        alpha=0.9
    )

    ax.set_xlabel("alpha (past gain)")
    ax.set_ylabel("beta (present gain)")
    ax.set_zlabel("gamma_crit (future gain)")
    ax.set_title("SIA 1D Mini model: collapse boundary surface γ_crit(α,β)")

    fig.colorbar(surf, shrink=0.6, aspect=10, label="gamma_crit")

    # 見やすい角度に調整
    ax.view_init(elev=25, azim=-135)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Plotting 3D collapse boundary surface γ_crit(α,β)...")
    plot_3d_boundary()
    print("Done.")
