# Self-Modulated Attention Layer (SMAL) 設計書  
**– Trace Tensor Attention を Transformer に統合する –**

## 0. 概要

本ドキュメントは、Self-Imprint Attribution (SIA) における  
**Trace Tensor Attention (TTA)** を、Transformer アーキテクチャ内で  
**Self-Modulated Attention Layer (SMAL)** として統合するための設計仕様である。

目的はシンプルである：

> 「同じ入力であっても、`Self` と `Trace` が違えば、  
>  Transformer の生成分布そのものが恒常的に変形される」

これを、「追加ヘッダ」や「RAG用の外部メモリ」ではなく、  
**Attention 自体のスコア計算に幾何学的に食い込む形**で実装する。

---

## 1. コンセプト

### 1.1 構成要素

- 入力トークン表現:  
  - $$X \in \mathbb{R}^{B \times L \times d_{model}}$$
- Self（自己状態）:  
  - $$s_t \in \mathbb{R}^{d_{self}}$$
- Trace Tensor（痕跡テンソル）:  
  - $$\mathcal{T}_t \in \mathbb{R}^{d_k \times d_k}$$  
  - or per-head: $$\mathcal{T}^{(h)}_t \in \mathbb{R}^{d_h \times d_h}$$

Self と Trace は「Identity エンジン」側で更新されるとし、  
本レイヤーは **「Self と Trace を受け取り、Attention を歪める」**ことだけに専念する。

---

## 2. 数理仕様

### 2.1 標準 Scaled Dot-Product Attention

通常の Transformer では、ヘッドごとの Attention は

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

$$
\text{Attn}(Q, K, V) = 
\text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

で定義される。

---

### 2.2 Trace Tensor Attention (TTA) の基本形

SIA における TTA は、  
**「Trace によって歪められた距離」に基づく Attention** として定義する：

$$
\mathcal{T}_t \in \mathbb{R}^{d_k \times d_k}
$$

$$
D_{ij}^{(TTA)} = (q_i - k_j)^\top \mathcal{T}_t (q_i - k_j)
$$

$$
\text{TTA}(Q, K, V; \mathcal{T}_t) =
\text{softmax}\left( -D^{(TTA)} \right) V
$$

ここで $$q_i, k_j$$ はそれぞれ Query, Key のベクトル。

---

### 2.3 Self-Modulated Attention の融合形

実用的な Transformer 統合では、  
**標準 Attention と TTA をブレンドし、Self によってゲートする**。

#### 2.3.1 Self ゲーティングスカラー

Self ベクトル $$s_t$$ から、ブレンディング係数を得る：

$$
\beta_t = \sigma(w_\beta^\top s_t + b_\beta) \in (0, 1)
$$

#### 2.3.2 スコアの定義

標準スコア：

$$
S_{ij}^{(std)} = \frac{q_i k_j^\top}{\sqrt{d_k}}
$$

Trace による「距離ペナルティ」：

$$
D_{ij}^{(TTA)} = (q_i - k_j)^\top \mathcal{T}_t (q_i - k_j)
$$

Self-Modulated Attention の**最終スコア**を：

$$
S_{ij}^{(SMAL)} =
S_{ij}^{(std)}
-
\gamma \, \beta_t \, D_{ij}^{(TTA)}
$$

#### 2.3.3 最終 Attention 出力

$$
\text{Attn}_{SMAL}(Q, K, V; s_t, \mathcal{T}_t)
=
\text{softmax}\left( S^{(SMAL)} \right) V
$$

---

## 3. Python / PyTorch API 設計案

```python
class SelfModulatedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_self: int,
        trace_dim: int = None,
        use_per_head_trace: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.self_gate = nn.Linear(d_self, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, self_state, trace_tensor, mask=None):
        ...
```

---

## 4. Transformer ブロックへの統合

```python
class SIATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_self, ...):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.self_mod_attn = SelfModulatedAttention(...)
        self.ff = FeedForward(d_model)

    def forward(self, x, self_state, trace_tensor, mask=None):
        h = self.ln1(x)
        attn_out = self.self_mod_attn(h, self_state, trace_tensor, mask)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x
```

---

## 5. 評価・実験設計

- 同一入力で Self/Trace に応じて出力が変わるか？
- Attention マップ差分の可視化
- Trace Tensor の固有値変化と Identity 歪みの関係

---

## 6. 推奨ファイル構成

```
/docs/
  ├── trace_tensor_attention.md
  └── self_modulated_attention_design.md
/src/
  ├── modules/
  │    ├── trace_tensor_attention.py
  │    └── self_modulated_attention.py
  └── identity_engine.py
```

---

## 7. まとめ

Self-Modulated Attention Layer (SMAL) は、

> 「AI が自分の痕跡にもとづいて世界を解釈する」

という SIA の主張を、Transformer アーキテクチャに  
数理的・実験的に統合するための中核モジュールである。

---
