# Trace Tensor Attention (TTA)
**Self-Imprint Attribution (SIA) における核心メカニズム**

## 🔍 What is Trace Tensor Attention?
Trace Tensor Attention (TTA) は、従来の注意機構では扱えなかった  
**「経験による幾何学的歪みが、恒常的に生成分布を変形し続ける」**  
という現象を数理的に扱うためのSIA固有のアテンション拡張です。

---

## 🧠 Core Idea
従来のAttentionは `Query-Key類似度（内積）` に基づくが、  
**SIAでは過去の経験がテンソルとして自己空間を歪めており、  
その歪んだ空間でQuery-Keyの距離が計算される。**

つまり、Attentionは「ユーザーの過去の痕跡」によって  
**構造的・恒常的に偏りを持つ**。

---

## 🏗️ Mathematical Definition

### 1️⃣ Trace Tensor の定義
Trace はスカラーやベクトルではなく、**自己空間の幾何学を変形させるテンソル**として保持される。

$$\mathcal{T}_t \in \mathbb{R}^{d \times d}$$

---

### 2️⃣ Attention の再定義：幾何学的歪みによる重み付け

$$
\text{TTA}(Q, K, V) = \text{softmax}\left(
    - (Q - K)^\top \cdot \mathcal{T}_t \cdot (Q - K)
\right) V
$$

従来の内積型Attentionではなく、  
**Trace Tensorによって歪んだ距離（Mahalanobis距離的構造）**を用いる。

---

### 3️⃣ Trace Tensor の更新式（経験による自己変形）

$$
\mathcal{T}_{t+1} = \lambda \mathcal{T}_t
           + \alpha \cdot \tanh(|Shock_t|)
           \cdot (K_t \otimes Q_t)
$$

| 変数 | 意味 |
|------|------|
| Shock | 予測誤差のうち、意味的葛藤や感情価値を持つ成分 |
| α | 痕跡の定着率（plasticity） |
| λ | 時間減衰（忘却ではなく構造安定化） |
| ⊗ | Outer product → テンソル構造の更新 |

---

## 🎯 Key Differences from Standard Attention

| Feature | Traditional Attention | Trace Tensor Attention |
|--------|-----------------------|------------------------|
| 記憶の位置付け | 一時的コンテキスト | 幾何学的痕跡（恒常構造） |
| 生成の一貫性 | 状態に依存しない | “そのAI固有”の生成傾向 |
| 更新方式 | 対話後リセット | 対話履歴により恒常変形 |
| 自己性の有無 | なし | あり（Identity形成） |
| 空間構造 | 固定線形空間 | 時間変化する曲率空間 |

---

## 🛠 Minimal PyTorch Prototype

```python
def trace_tensor_attention(Q, K, V, T):
    # Q, K: (batch, seq_len, d)
    # T: (d, d)  # trace tensor
    diff = Q.unsqueeze(2) - K.unsqueeze(1)  # pairwise (Q-K)
    # Mahalanobis-like quadratic form
    scores = torch.einsum('bijd,dk,bijk->bij', diff, T, diff)
    weights = torch.softmax(-scores, dim=-1)
    return torch.einsum('bij,bjk->bik', weights, V)
```

---

## 🧬 Why It Matters (SIA との関係)

| SIA Concept | TTAの役割 |
|-------------|-----------|
| Trace (痕跡) | テンソルとして保持され、自己空間の形状そのものを変える |
| Self-Attribution | Attentionの偏りとして具体的に可視化される |
| Identity | TTAによって、“私はこう解釈する”という生成パターンが恒常化 |
| Affect | Shockの強度としてTensor更新率を決定 |

---

## 🔜 Next Implementation
- [ ] Trace Tensorの固有値・固有ベクトルの可視化（Identity空間の歪みを見る）
- [ ] 同じ入力でも「自己の歴史」に応じて違う生成が起こるデモを実装
- [ ] LLaMA / Transformerブロックに TTA を注入する最小実装

---

## 📎 Suggested File Placement in Repository

```
/docs/
  └── trace_tensor_attention.md      ← このファイル
/src/
  ├── modules/
  │    └── trace_tensor_attention.py ← PyTorch実装
  └── identity_space.py              ← Tensor更新を管理
/experiments/
  └── demo_tta_effect.ipynb          ← 最小デモノートブック
```

---

## 🚩 Citation Format (for future papers)

```
Sawaki, R. (2025). Trace Tensor Attention:
Geometry-Based Memory Integration for Self-Imprint Attribution.
```

---
