# src/identity/encoder/self_aware_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAwareEncoder(nn.Module):
    """
    LLaMAから Self-space 入力 u_t を抽出するための Encoder。
    - hidden state (全token) を取得
    - self_center に基づく注意で「自分に関係する部分」を抽出
    - Self-space 次元へ射影して u_t を返す
    """

    def __init__(self, llama_model, hidden_dim, self_dim):
        super().__init__()
        self.llama = llama_model  # HF LLaMAモデル想定
        self.hidden_dim = hidden_dim
        self.self_dim = self_dim

        # Self-space への写像
        self.project_to_self = nn.Linear(hidden_dim, self_dim)

        # Self-center を LLaMA空間に写す
        self.self_to_llama = nn.Linear(self_dim, hidden_dim)

    def forward(self, input_text: str, self_center: torch.Tensor):
        """
        input_text : str
        self_center : (self_dim,)
        return : u_t (self_dim,)
        """

        # 1) LLaMA hidden states 取得
        outputs = self.llama(input_text, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        hidden = hidden.squeeze(0)  # (seq_len, hidden_dim)

        # 2) Self-center を LLaMA埋め込み空間へ
        self_proj = self.self_to_llama(self_center)  # (hidden_dim,)

        # 3) tokenごとに自己関係性スコアを計算
        scores = torch.matmul(hidden, self_proj)  # (seq_len,)
        weights = F.softmax(scores / 0.1, dim=0)  # 温度 0.1（敏感に自己選択）
        pooled = torch.sum(weights.unsqueeze(-1) * hidden, dim=0)

        # 4) Self-space へ写像
        u_t = self.project_to_self(pooled)  # (self_dim,)

        return u_t
