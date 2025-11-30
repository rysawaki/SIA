"""
exp_self_conditioned_attention.py

Self-conditioning が Transformer の Attention と Query 表現に
どのような影響を与えるかを最小実験で検証するスクリプト。

出力：
 - outputs/query_deformation.png
 - outputs/attention_comparison.png
"""

import torch
import os
import matplotlib.pyplot as plt

from src.transformer.mini_transformer import MiniTransformer
from src.utils.visualize_query_deformation import visualize_query_deformation


def run_experiment():
    torch.manual_seed(42)

    # 1. モデル準備
    model = MiniTransformer(
        vocab_size=100,
        d_model=128,
        n_layers=3,
        n_heads=4,
        d_ff=256,
        max_self_axes=6,
        dropout=0.0
    )
    model.eval()

    # 2. 入力トークン生成（ランダム）
    seq_len, batch_size = 6, 1
    input_ids = torch.randint(5, 99, (seq_len, batch_size))

    # 3. Selfがない状態でフォワード
    hidden_no_self, attn_no_self = model(input_ids, need_attention=True)

    # 4. Trace（最後のトークンの隠れ表現）でSelfを更新
    trace_vec = hidden_no_self[-1, 0]
    model.self_space.update(
        trace=trace_vec,
        shock=0.9,
        affect=0.8
    )

    print("\n[SelfSpace Metrics after update:]")
    print(model.self_space.metrics())

    # 5. Self更新後に再フォワード
    hidden_self, attn_self = model(input_ids, need_attention=True)

    # 6. Query変形の可視化
    os.makedirs("../../experiments/outputs/03", exist_ok=True)
    visualize_query_deformation(
        Q=hidden_no_self[-1, 0],
        Q_cond=hidden_self[-1, 0],
        axes=model.self_space.axes[:model.self_space.num_active],
        save_path="../outputs/03/03_query_deformation.png"
    )

    # 7. Attention比較の可視化
    fig, axes = plt.subplots(2, attn_no_self.shape[1], figsize=(15, 6))

    for h in range(attn_no_self.shape[1]):
        axes[0, h].imshow(attn_no_self[0, h].detach().numpy(), cmap='viridis')
        axes[0, h].set_title(f"Before Self - Head {h}")

        axes[1, h].imshow(attn_self[0, h].detach().numpy(), cmap='viridis')
        axes[1, h].set_title(f"After Self - Head {h}")

    plt.tight_layout()
    plt.savefig("../../experiments/outputs/03/03_attention_comparison.png", dpi=300)
    plt.show()

    print("\nSaved outputs to:")
    print(" - outputs/03/03_query_deformation.png")
    print(" - outputs/03/03_attention_comparison.png")


if __name__ == "__main__":
    run_experiment()
