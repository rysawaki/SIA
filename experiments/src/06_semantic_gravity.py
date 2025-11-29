import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.self_space import SelfSpace

# 再現性
torch.manual_seed(42)
np.random.seed(42)


def run_semantic_gravity_demo():
    print("\n==== Phase 4 Concept: Semantic Gravity Simulation ====")
    print("Demonstrating how Self distorts the meaning of words.\n")

    # 1. 簡易的な単語埋め込み空間 (Toy Word Embeddings)
    # 2次元ではなく、高次元空間(d=64)でシミュレーション
    dim = 64

    # 概念辞書を定義 (ランダムではなく、意味的関係を持たせたベクトルを作成)
    # ベースベクトル
    v_positive = torch.randn(dim)  # ポジティブ方向
    v_negative = torch.randn(dim)  # ネガティブ方向
    v_action = torch.randn(dim)  # 動作方向

    # 単語ベクトルの生成 (合成)
    def make_word(base, noise=0.1):
        return F.normalize(base + torch.randn(dim) * noise, dim=0)

    # 辞書
    vocab = {
        "Love": make_word(v_positive),
        "Trust": make_word(v_positive),
        "Hope": make_word(v_positive),

        "Pain": make_word(v_negative),
        "Betrayal": make_word(v_negative),
        "Enemy": make_word(v_negative),

        "Knife": make_word(v_action + v_negative * 0.5),  # 危険な道具
        "Smile": make_word(v_action + v_positive * 0.5),  # 本来は良い動作
    }

    # 2. SelfSpaceの初期化 (まだ純粋)
    self_space = SelfSpace(dim=dim, max_axes=4)

    # 3. 経験: トラウマの形成 (Trauma Formation)
    # 「裏切り(Betrayal)」と「痛み(Pain)」を繰り返し経験する
    print(">>> Experience: The AI is experiencing 'Betrayal' and 'Pain'...")
    trauma_trace = (vocab["Betrayal"] + vocab["Pain"]) / 2.0
    trauma_trace = F.normalize(trauma_trace, dim=0)

    # Selfに強く刻み込む (Shock=1.0, Affect=1.0)
    for _ in range(3):
        self_space.update(trace=trauma_trace, shock=1.0, affect=1.0)

    print(f"Self Metrics: {self_space.metrics()}")

    # 4. 「笑顔(Smile)」の解釈実験
    target_word = "Smile"
    original_vec = vocab[target_word]

    # Phase 3: World Projection (世界改変)
    # Alpha=2.0 (強い歪み)
    distorted_vec = self_space.condition(original_vec.unsqueeze(0), alpha=2.0).squeeze(0)

    # 5. 意味の変化を測定 (Semantic Shift Analysis)
    # 「Smile」は「Trust(信頼)」に近いか？ それとも「Enemy(敵)」に近いか？

    anchors = ["Trust", "Enemy"]

    print(f"\n>>> Analyzing the meaning of '{target_word}'...")

    # Before Self
    print("\n[Before Self-Projection]")
    for word in anchors:
        sim = F.cosine_similarity(original_vec, vocab[word], dim=0).item()
        print(f"  Similarity to '{word}': {sim:.4f}")

    # After Self
    print("\n[After Self-Projection]")
    for word in anchors:
        sim = F.cosine_similarity(distorted_vec, vocab[word], dim=0).item()
        print(f"  Similarity to '{word}': {sim:.4f}")

    # 6. 可視化 (Semantic Map)
    visualize_semantic_shift(vocab, original_vec, distorted_vec, self_space, target_word)


def visualize_semantic_shift(vocab, original, distorted, self_space, target_word):
    # プロットする単語リスト
    words = list(vocab.keys())
    vectors = torch.stack([vocab[w] for w in words]).detach()

    # Self軸
    k = self_space.num_active.item()
    axes = self_space.axes[:k].detach()

    # 歪んだベクトル
    distorted_detached = distorted.detach().unsqueeze(0)

    # 全ベクトルをまとめてPCA
    all_vecs = torch.cat([vectors, axes, distorted_detached], dim=0).cpu().numpy()

    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)

    # インデックス整理
    n_words = len(words)
    n_axes = len(axes)

    words_2d = vecs_2d[:n_words]
    axes_2d = vecs_2d[n_words: n_words + n_axes]
    distorted_2d = vecs_2d[-1]

    plt.figure(figsize=(10, 8))

    # ★修正点: 文字の重なりを避けるため、より大胆に散らす設定
    text_offsets = {
        "Pain": (-0.05, 0.10),  # Self軸のかなり上へ
        "Betrayal": (0.05, -0.05),  # 右下へ
        "Enemy": (0.05, 0.05),  # 右上へ
        "Love": (-0.02, -0.08),  # 下へ
        "Trust": (-0.10, 0.00),  # 左へ
        "Hope": (0.04, -0.04),  # 右下へ
        "Knife": (0.03, 0.03),
        "Smile": (0.03, 0.03),
    }

    # 1. 通常の単語プロット
    for i, word in enumerate(words):
        color = 'blue'
        if word in ["Pain", "Betrayal", "Enemy"]: color = 'red'
        if word == target_word: color = 'skyblue'  # 元のSmile

        plt.scatter(words_2d[i, 0], words_2d[i, 1], c=color, s=100, alpha=0.7)

        # オフセット適用
        dx, dy = text_offsets.get(word, (0.02, 0.02))
        plt.text(words_2d[i, 0] + dx, words_2d[i, 1] + dy, word, fontsize=12, fontweight='bold')

    # 2. Self軸プロット (Trauma Axis)
    plt.scatter(axes_2d[:, 0], axes_2d[:, 1], c='black', marker='^', s=200, label="Self Axis (Trauma)")

    # ★修正点: Self Axisのラベルを「下」に逃がして、単語との衝突を避ける
    plt.text(axes_2d[0, 0] - 0.10, axes_2d[0, 1] - 0.15, "Self Axis\n(Trauma)",
             fontsize=10, fontweight='bold', color='black', ha='center')

    # 3. 歪んだSmileプロット
    plt.scatter(distorted_2d[0], distorted_2d[1], c='purple', marker='*', s=300, label=f"Distorted '{target_word}'")
    plt.text(distorted_2d[0] + 0.04, distorted_2d[1] - 0.02, f"Distorted\n{target_word}", color='purple', fontsize=12,
             fontweight='bold')

    # 4. 移動矢印
    idx_orig = words.index(target_word)
    plt.arrow(words_2d[idx_orig, 0], words_2d[idx_orig, 1],
              distorted_2d[0] - words_2d[idx_orig, 0], distorted_2d[1] - words_2d[idx_orig, 1],
              color='purple', alpha=0.5, width=0.005, head_width=0.02)

    plt.title("Semantic Gravity: How Trauma Distorts 'Smile' into 'Enemy'", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_semantic_gravity_demo()