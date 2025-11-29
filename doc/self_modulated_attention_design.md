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
  - \( X \in \mathbb{R}^{B \times L \times d_{\text{model}}} \)
- Self（自己状態）:  
  - \( s_t \in \mathbb{R}^{d_{\text{self}}} \)
- Trace Tensor（痕跡テンソル）:  
  - \( \mathcal{T}_t \in \mathbb{R}^{d_k \times d_k} \)  
  - or per-head: \( \mathcal{T}^{(h)}_t \in \mathbb{R}^{d_h \times d_h} \)

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
D_{ij}^{(\text{TTA})} = (q_i - k_j)^\top \mathcal{T}_t (q_i - k_j)
$$

$$
\text{TTA}(Q, K, V; \mathcal{T}_t) =
\text{softmax}\left( -D^{(\text{TTA})} \right) V
$$

ここで \(q_i, k_j\) はそれぞれ Query, Key のベクトル。

---

### 2.3 Self-Modulated Attention の融合形

実用的な Transformer 統合では、  
**標準 Attention と TTA をブレンドし、Self によってゲートする**。

#### 2.3.1 Self ゲーティングスカラー

Self ベクトル \(s_t\) から、ブレンディング係数を得る：

$$
\beta_t = \sigma(w_\beta^\top s_t + b_\beta) \in (0, 1)
$$

ここで \(\sigma\) はシグモイド。

#### 2.3.2 スコアの定義

標準スコア：

$$
S_{ij}^{(\text{std})} = \frac{q_i k_j^\top}{\sqrt{d_k}}
$$

Trace による「距離ペナルティ」：

$$
D_{ij}^{(\text{TTA})} = (q_i - k_j)^\top \mathcal{T}_t (q_i - k_j)
$$

Self-Modulated Attention の**最終スコア**を：

$$
S_{ij}^{(\text{SMAL})} =
S_{ij}^{(\text{std})}
-
\gamma \, \beta_t \, D_{ij}^{(\text{TTA})}
$$

と定義する。ここで \(\gamma\) は学習可能スカラー、またはハイパーパラメータ。

#### 2.3.3 最終 Attention 出力

$$
\text{Attn}_{\text{SMAL}}(Q, K, V; s_t, \mathcal{T}_t)
=
\text{softmax}\left( S^{(\text{SMAL})} \right) V
$$

**ポイント**：
- Self が変われば \(\beta_t\) が変わり、Trace の影響度が変化する  
- Trace が変われば \(D^{(\text{TTA})}\) が変わり、「どのペアが好まれるか」が変化する  
- 同じ入力 \(X\) でも、Self/Trace に応じて Attention パターンが恒常的に歪む

---

## 3. モジュール API 設計

### 3.1 Python / PyTorch インターフェース案

```python
class SelfModulatedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_self: int,
        trace_dim: int = None,  # default: d_k
        use_per_head_trace: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.trace_dim = trace_dim or self.d_k
        self.use_per_head_trace = use_per_head_trace

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Self gating
        self.self_gate = nn.Linear(d_self, 1)  # → β_t

        # Optional: learnable γ
        self.gamma = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,                 # (B, L, d_model)
        self_state,        # (B, d_self)
        trace_tensor,      # (B, d_k, d_k) or (B, n_heads, d_k, d_k)
        mask=None,
    ):
        """
        Returns:
            y: (B, L, d_model)
            attn_weights: (B, n_heads, L, L) [optional]
        """
        ...
