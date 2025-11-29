import sys
import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.self_space import SelfSpace
from src.mini_transformer import MiniTransformer

def run_experiment_over_time():
    torch.manual_seed(42)
    seq_len = 6
    vocab_size = 50
    num_steps = 5

    model = MiniTransformer(
        vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4, d_ff=64
    )
    
    # バッチサイズ1の入力
    input_ids = torch.randint(0, vocab_size, (1, seq_len)) 

    attn_history = []
    query_history = []

    for step in range(num_steps):
        # Self更新前のForward
        hidden, attn = model(input_ids, need_attention=True)
        attn_history.append(attn.detach().cpu()) # (B, Heads, L, L)

        # SelfAttention層から保存されたQueryを取得
        # 構造: model.layers[-1] (SelfConditionedEncoderLayer) -> .self_attn -> .last_queries
        last_queries = model.layers[-1].self_attn.last_queries # (B, L, E)
        query_history.append(last_queries.detach().cpu())

        # Selfの更新（ダミーのTraceとして、現在のQueryの平均ベクトルを使用）
        # 実際は隠れ状態(hidden)などを使うことが多い
        trace_vec = last_queries.mean(dim=1).squeeze(0) # (E,)
        shock = torch.rand(1).item() * 0.5 + 0.1
        
        # モデル内のself_spaceを直接更新
        model.self_space.update(trace=trace_vec, shock=shock, affect=1.0)

        print(f"[Step {step+1}] Self Metrics: {model.self_space.metrics()}")

    visualize_attention_over_time(attn_history)
    visualize_query_trajectory(query_history)

def visualize_attention_over_time(attn_history):
    num_steps = len(attn_history)
    fig, axes = plt.subplots(1, num_steps, figsize=(3*num_steps, 3))
    if num_steps == 1: axes = [axes]
    
    for i, attn in enumerate(attn_history):
        # Head 0, Batch 0 を表示
        ax = axes[i]
        ax.imshow(attn[0, 0].numpy(), cmap='viridis')
        ax.set_title(f"Step {i}")
        ax.axis('off')

    plt.suptitle("Attention Map Evolution (Head 0)")
    plt.tight_layout()
    plt.show()

def visualize_query_trajectory(query_history):
    # query_history: list of (B, L, E)
    # 全ステップの全トークンのQueryをPCAで投影
    all_queries = torch.cat(query_history, dim=0) # (Steps, L, E) assuming B=1
    steps, seq, dim = all_queries.shape
    flat_q = all_queries.reshape(-1, dim).numpy() # (Steps*L, dim)

    pca = PCA(n_components=2)
    q_2d = pca.fit_transform(flat_q)
    q_2d = q_2d.reshape(steps, seq, 2)

    plt.figure(figsize=(8, 6))
    
    # 時間変化をトークンごとに追跡
    colors = plt.cm.jet(torch.linspace(0, 1, seq).numpy())
    for t in range(seq):
        traj = q_2d[:, t, :] # (Steps, 2)
        plt.plot(traj[:, 0], traj[:, 1], marker='.', label=f'Token {t}', color=colors[t], alpha=0.7)
        # Start point
        plt.scatter(traj[0, 0], traj[0, 1], marker='o', s=50, color=colors[t])
        # End point
        plt.scatter(traj[-1, 0], traj[-1, 1], marker='x', s=50, color=colors[t])

    plt.title("Query Trajectory in Self Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment_over_time()