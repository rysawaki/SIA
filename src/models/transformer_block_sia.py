# ============================
# file: transformer_block_sia.py
# Transformer block with Self-conditioned attention
# ============================

import torch
import torch.nn as nn

from src.identity.core.self_space import SelfSpace
from sia_attention import SelfConditionedMultiheadAttention


class SIATransformerBlock(nn.Module):
    """
    通常の Transformer Block に SelfSpace ベースの Attention を組み込んだ版。

    特徴:
    - Query が SelfSpace を通過し、幾何学的に歪んだ上で Attention に入る。
    - SelfSpace.update() は外部から明示的に呼び出して状態進化させる。
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, device="cpu"):
        super().__init__()

        # --- SelfSpace instance (shared across blocks if needed) ---
        self.self_space = SelfSpace(dim=dim, device=device)

        # --- Standard Transformer components ---
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfConditionedMultiheadAttention(
            dim=dim,
            num_heads=num_heads,
            self_space=self.self_space,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, mask=None):
        """
        x: (B,T,D)
        """
        # === Self-Conditioned Attention ===
        h = self.ln1(x)
        h = self.attn(h, mask=mask)
        x = x + h  # residual

        # === FFN ===
        h2 = self.ln2(x)
        h2 = self.mlp(h2)
        x = x + h2  # residual

        return x
