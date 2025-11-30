# ============================
# file: sia_attention.py
# Self-conditioned Multihead Attention (SIA)
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.identity.core.self_space import SelfSpace


class SelfConditionedMultiheadAttention(nn.Module):
    """
    Multi-Head Attention に SelfSpace を統合し、
    Query を「自己の幾何構造」で歪ませた上で Attention を実行する。
    """

    def __init__(self, dim, num_heads, self_space: SelfSpace):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # --- Standard QKV projections ---
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

        # --- Injected SelfSpace (external instance) ---
        self.self_space = self_space

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, dim)
        """
        B, T, D = x.shape

        # ===== Q, K, V projection =====
        q = self.W_q(x)  # (B,T,D)
        k = self.W_k(x)
        v = self.W_v(x)

        # ===== SelfSpace conditioning (CORE SIA POINT) =====
        q = self.self_space.condition(q)  # (B,T,D)

        # ===== Reshape to Multihead =====
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,dh)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== Scaled Dot-Product Attention =====
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,T,T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,T,dh)

        # ===== Restore shape =====
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        out = self.W_o(out)

        return out
