# src/self_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfConditionedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, self_space,
                 dropout: float = 0.0, bias: bool = True, alpha: float = 0.6):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.self_space = self_space
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 分析用のメンバ変数
        self.last_queries = None

    def _shape(self, x: torch.Tensor, L: int, B: int):
        x = x.view(L, B, self.num_heads, self.head_dim)
        x = x.permute(1, 2, 0, 3).reshape(B * self.num_heads, L, self.head_dim)
        return x

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        L, B, E = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # =======================================================
        # Phase 3: Subjective Reality (主観的現実の構築)
        # Selfは視点(Query)だけでなく、世界(Key)の意味構造も歪める
        # =======================================================

        # 1. 視点の歪み (Query Deformation)
        Q_flat = Q.permute(1, 0, 2).reshape(B * L, E)
        Q_cond = self.self_space.condition(Q_flat, alpha=self.alpha)
        Q_cond = Q_cond.reshape(B, L, E).permute(1, 0, 2)

        # 可視化のために保存
        self.last_queries = Q_cond.permute(1, 0, 2).detach()

        # 2. 世界の再構成 (Key Deformation) - NEW!
        # 客観的な入力 K を、Selfの軸に沿って引き寄せる(投影する)
        K_flat = K.permute(1, 0, 2).reshape(B * L, E)
        K_cond = self.self_space.condition(K_flat, alpha=self.alpha)
        K_cond = K_cond.reshape(B, L, E).permute(1, 0, 2)

        # =======================================================

        q = self._shape(Q_cond, L, B)
        k = self._shape(K_cond, L, B) # 歪んだ世界 K_cond を使用
        v = self._shape(V, L, B)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).expand(B, self.num_heads, L)
            kpm = kpm.reshape(B * self.num_heads, 1, L)
            scores = scores.masked_fill(kpm, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.reshape(B, self.num_heads, L, self.head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(L, B, E)
        attn_output = self.out_proj(attn_output)

        # 戻り値用整形 (Batch, Head, L, L)
        attn_weights_reshaped = attn_weights.reshape(B, self.num_heads, L, L)

        return attn_output, attn_weights_reshaped


# EncoderLayerなどは変更なし
class SelfConditionedEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, self_space, dropout=0.1):
        super().__init__()
        self.self_attn = SelfConditionedMultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, self_space=self_space, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(x, attn_mask, key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x, attn_weights