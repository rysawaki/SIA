import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os

# ==============================
# パラメータ設定
# ==============================

DEVICE = "cpu"
DIM = 2
STEPS = 200          # 時系列ステップ数
ALPHA = 0.05         # GrowthKernelの学習率
EPS = 1e-6

# 出力ディレクトリ
OUTPUT_DIR = os.path.join("..", "..", "experiments", "outputs", "13", "frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# GrowthKernel の定義
# ==============================

def novelty_gain(delta_norm: torch.Tensor) -> torch.Tensor:
    """
    g(||delta||): 新規性が大きいほど成長を増やす。
    """
    return torch.tanh(delta_norm)


def saturation_factor(I: torch.Tensor) -> torch.Tensor:
    """
    h(||I||_F): 既に十分育っているなら成長を抑える。
    """
    frob = torch.norm(I, p="fro")
    return torch.sigmoid(1.0 - frob)


def growth_kernel(delta: torch.Tensor, I: torch.Tensor, A_t: float) -> torch.Tensor:
    """
    GrowthKernel(delta, I_t, A_t)
    delta: (2,)  Selfとの差
    I:     (2,2) current Imprint / 気づきテンソル
    A_t:   scalar, Attribution / 意味の自己関連度
    """
    delta_norm = torch.norm(delta) + EPS
    dir_tensor = torch.outer(delta, delta) / (delta_norm ** 2)   # 方向（単位テンソル）

    g = novelty_gain(delta_norm)
    h = saturation_factor(I)

    update = ALPHA * A_t * g * h * dir_tensor
    return update


# ==============================
# イベント生成（Selfに対する入力）
# ==============================

def sample_event(t: int) -> torch.Tensor:
    """
    デモ用：時間によって少しずつ分布をずらす。
    前半はx軸方向、後半はy軸方向のイベントが増えるようにする。
    """
    if t < STEPS // 2:
        mean = torch.tensor([2.0, 0.5])
    else:
        mean = torch.tensor([0.0, 2.0])

    cov = torch.tensor([[0.5, 0.0],
                        [0.0, 0.5]])
    L = torch.linalg.cholesky(cov)
    z = torch.randn(2)
    return mean + L @ z


# ==============================
# 楕円の描画ユーティリティ
# ==============================

def plot_state(I: torch.Tensor, events: np.ndarray, step: int, save_path: str):
    """
    I: (2,2) 気づきテンソル
    events: (N, 2) これまでに観測したイベント
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # イベント点の描画
    if len(events) > 0:
        ax.scatter(events[:, 0], events[:, 1], alpha=0.4, s=10)

    # 原点（Self）を描画
    ax.scatter([0.0], [0.0], marker="x")

    # I の固有分解
    I_np = I.detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(I_np)

    # 数値誤差で負になることがあるのでクリップ
    eigvals = np.clip(eigvals, 1e-6, None)

    # 楕円の大きさスケール
    scale = 2.0
    width = 2 * scale * np.sqrt(eigvals[1])
    height = 2 * scale * np.sqrt(eigvals[0])

    # 回転角（度）
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    ellipse = Ellipse(
        xy=(0.0, 0.0),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2
    )
    ax.add_patch(ellipse)

    ax.set_title(f"GrowthKernel Demo - step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ==============================
# メインループ
# ==============================

def run_growth_kernel_demo():
    # Selfは原点に固定（s_t = 0）
    s = torch.zeros(DIM, device=DEVICE)

    # Imprint（気づきテンソル）の初期値
    I = torch.zeros(DIM, DIM, device=DEVICE)

    events_list = []

    for t in range(STEPS):
        # 1. イベントをサンプル
        u_t = sample_event(t)
        delta = u_t - s

        # 2. Attribution / 意味関連度（ここでは単純に 0〜1 の乱数にしておく）
        A_t = float(torch.rand(()))  # 将来はSelf-Spaceから計算しても良い

        # 3. GrowthKernel によって I を更新
        I = I + growth_kernel(delta, I, A_t)

        # 数値誤差対策で対称化
        I = 0.5 * (I + I.T)

        # イベント履歴に追加（可視化用）
        events_list.append(u_t.numpy())

        # 4. 可視化フレームを保存
        save_path = os.path.join(OUTPUT_DIR, f"frame_{t:04d}.png")
        plot_state(I, np.vstack(events_list), t, save_path)

    print(f"Saved frames to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_growth_kernel_demo()
