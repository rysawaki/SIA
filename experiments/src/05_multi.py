import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mini_transformer import MiniTransformer
# SelfSpaceの定義が必要なためimport（型チェック等で使う場合）
from src.self_space import SelfSpace

# 再現性
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. 幾何学的 Shock / Affect の定義
# ==========================================

def compute_geometric_shock(trace, self_space):
    if self_space.num_active.item() == 0:
        return 0.5
    k = self_space.num_active.item()
    axes = self_space.axes[:k]
    dists = torch.norm(axes - trace.unsqueeze(0), dim=1)
    min_dist = dists.min().item()
    shock = np.clip(min_dist / 2.0, 0.0, 1.0)
    return shock


def compute_geometric_affect(trace, self_space):
    if self_space.num_active.item() == 0:
        return 1.0
    k = self_space.num_active.item()
    axes = self_space.axes[:k]
    sims = F.cosine_similarity(trace.unsqueeze(0), axes, dim=1)
    max_sim = sims.max().item()
    affect = np.clip(max_sim, 0.1, 1.0)
    return affect


# ==========================================
# 2. 実験設定: 差分可視化によるSIAの証明
# ==========================================

def mutate_input_one_token(base_seq, vocab_range=(0, 10)):
    seq = base_seq.clone()
    idx = torch.randint(0, seq.shape[0], (1,)).item()
    new_token = torch.randint(vocab_range[0], vocab_range[1], (1,)).item()
    seq[idx, 0] = new_token
    return seq


def run_geometric_identity_experiment():
    print("\n==== Geometric Self Identity Experiment (Differential Analysis) ====")

    vocab_size = 50
    model = MiniTransformer(vocab_size=vocab_size, d_model=32, n_layers=2, pad_token_id=-1)

    # Selfの影響を強調
    model.layers[-1].self_attn.alpha = 2.0
    self_space = model.self_space

    # --- シナリオ ---
    base_A = torch.randint(0, 15, (6, 1))
    base_B = torch.randint(35, 50, (6, 1))

    # Experience Sequence
    inputs_A1 = base_A.clone()
    inputs_A2 = mutate_input_one_token(base_A)  # Update狙い
    inputs_A3 = mutate_input_one_token(base_A)  # Update狙い
    inputs_B1 = base_B.clone()  # New Axis狙い
    inputs_B2 = mutate_input_one_token(base_B)  # New/Update狙い
    inputs_A4 = base_A.clone()  # Update (Return)

    experiences = [inputs_A1, inputs_A2, inputs_A3, inputs_B1, inputs_B2, inputs_A4]
    labels = ["A1(Init)", "A2(Same)", "A3(Same)", "B1(Shock)", "B2(Adapt)", "A4(Return)"]

    history_metrics = []
    traces = []
    attn_diffs = []  # 差分保存用

    print(f"{'Step':<5} | {'Type':<10} | {'Shock':<8} | {'Affect':<8} | {'Action'}")
    print("-" * 55)

    for i, inp in enumerate(experiences):
        # --- 比較実験: Selfなし vs Selfあり ---

        # 1. Selfなし (Control Group)
        # 一時的にSelfSpaceを無効化して推論
        original_axes = self_space.num_active.item()
        self_space.num_active.fill_(0)  # 無効化
        _, attn_base = model(inp, need_attention=True)
        self_space.num_active.fill_(original_axes)  # 復元

        # 2. Selfあり (Experimental Group)
        hidden, attn_self = model(inp, need_attention=True)

        # 差分計算 (Self - Base)
        # プラスならSelfが注目させた箇所、マイナスなら無視させた箇所
        diff = attn_self - attn_base
        attn_diffs.append(diff.detach())

        # Trace
        trace = hidden.mean(dim=0).mean(dim=0).detach()
        trace = F.normalize(trace, dim=0)
        traces.append(trace)

        # 3. Parameters
        shock = compute_geometric_shock(trace, self_space)
        affect = compute_geometric_affect(trace, self_space)

        # 4. Update
        prev_axes = self_space.num_active.item()
        self_space.update(trace=trace, shock=shock, affect=affect, sim_threshold=0.80)

        # 5. Log
        curr_axes = self_space.num_active.item()
        action = "New" if curr_axes > prev_axes else "Upd"
        if i == 0: action = "Init"

        print(f"{i + 1:<5} | {labels[i]:<10} | {shock:<8.4f} | {affect:<8.4f} | {action}")

        m = self_space.metrics()
        m.update({"shock": shock, "affect": affect})
        history_metrics.append(m)

    visualize_geometric_identity(history_metrics, traces, self_space, labels)
    visualize_differential_attention(attn_diffs, labels)


# ==========================================
# 3. 可視化関数群
# ==========================================

def visualize_geometric_identity(metrics, traces, self_space, labels):
    steps = range(len(metrics))
    shocks = [m['shock'] for m in metrics]
    affects = [m['affect'] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Metrics
    ax1.plot(steps, shocks, 'o-', label='Shock', color='red')
    ax1.plot(steps, affects, 's-', label='Affect', color='blue')
    ax1.set_title("Geometric Perception Dynamics")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Intensity")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PCA
    traces_tensor = torch.stack(traces)
    active_axes = self_space.axes[:self_space.num_active.item()].detach()
    all_vecs = torch.cat([traces_tensor, active_axes], dim=0).numpy()

    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)

    n_traces = len(traces)
    traces_2d = vecs_2d[:n_traces]
    axes_2d = vecs_2d[n_traces:]

    for i in range(n_traces):
        color = 'blue' if 'A' in labels[i] else 'red'
        ax2.scatter(traces_2d[i, 0], traces_2d[i, 1], c=color, s=100, alpha=0.6)
        ax2.text(traces_2d[i, 0], traces_2d[i, 1] + 0.02, str(i + 1), fontsize=9)
        if i > 0:
            ax2.arrow(traces_2d[i - 1, 0], traces_2d[i - 1, 1],
                      traces_2d[i, 0] - traces_2d[i - 1, 0], traces_2d[i, 1] - traces_2d[i - 1, 1],
                      color='gray', alpha=0.3, width=0.002)

    if len(axes_2d) > 0:
        ax2.scatter(axes_2d[:, 0], axes_2d[:, 1], c='green', marker='^', s=200, edgecolors='black', label='Self Axes')
        for i in range(len(axes_2d)):
            ax2.text(axes_2d[i, 0] + 0.05, axes_2d[i, 1], f"Ax{i + 1}", fontsize=12, color='green', fontweight='bold')

    ax2.set_title("Meaning Geometry (PCA)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_differential_attention(attn_diffs, labels):
    """
    Selfあり/なしの差分を表示する。
    赤色(Positive): Selfが注目させた場所
    青色(Negative): Selfが無視させた場所
    """
    num_steps = len(attn_diffs)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 3))
    if num_steps == 1: axes = [axes]

    # 差分の絶対値の最大をとって、0中心の対称なカラーマップにする
    max_val = 0.0
    for diff in attn_diffs:
        max_val = max(max_val, diff.abs().max().item())

    # 0除算回避
    if max_val < 1e-9: max_val = 1.0

    for i, diff in enumerate(attn_diffs):
        ax = axes[i]
        heatmap = diff[0, 0].cpu().numpy()

        # bwr (Blue-White-Red) or seismic を使用
        # 0が白、プラスが赤、マイナスが青
        im = ax.imshow(heatmap, cmap='bwr', vmin=-max_val, vmax=max_val)
        ax.set_title(f"{i + 1}. {labels[i]}")
        ax.axis('off')

    plt.suptitle("Differential Attention: What Self Adds to Reality")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_geometric_identity_experiment()