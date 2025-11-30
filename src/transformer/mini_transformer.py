# src/mini_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.identity.core.self_space import SelfSpace
from src.transformer.self_attention import SelfConditionedEncoderLayer


class SimpleEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def _shape(self, x, L, B):
        # Correct dimensions: (Batch*Head, Length, Dim)
        x = x.view(L, B, self.n_heads, self.head_dim)
        x = x.permute(1, 2, 0, 3).reshape(B * self.n_heads, L, self.head_dim)
        return x

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        L, B, E = x.shape
        q = self._shape(self.q_proj(x), L, B)
        k = self._shape(self.k_proj(x), L, B)
        v = self._shape(self.v_proj(x), L, B)

        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        if attn_mask is not None: scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(self.dropout(attn_weights), v)

        attn_output = attn_output.reshape(B, self.n_heads, L, self.head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(L, B, E)

        x = self.norm1(x + self.dropout(self.out_proj(attn_output)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=512,
                 max_self_axes=8, dropout=0.1, pad_token_id=0, device="cpu"):
        super().__init__()
        self.device = device
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.self_space = SelfSpace(dim=d_model, max_axes=max_self_axes, device=device)

        self.layers = nn.ModuleList([
            SimpleEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers - 1)
        ])
        self.layers.append(
            SelfConditionedEncoderLayer(d_model, n_heads, d_ff, self.self_space, dropout)
        )
        self.to(device)

    def forward(self, input_ids, attn_mask=None, need_attention=False):
        x = self.token_embed(input_ids).transpose(0, 1)  # (L, B, E)
        key_padding_mask = (input_ids == self.pad_token_id)

        final_attn = None
        for layer in self.layers:
            if isinstance(layer, SelfConditionedEncoderLayer):
                x, final_attn = layer(x, attn_mask, key_padding_mask)
            else:
                x = layer(x, attn_mask, key_padding_mask)

        if need_attention:
            return x.transpose(0, 1), final_attn
        return x.transpose(0, 1)