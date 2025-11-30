# src/identity/generation/self_conditioned_generator.py
# -*- coding: utf-8 -*-

"""
Self-conditioned Text Generation

役割:
    - ベースとなる LLM (AutoModelForCausalLM)
    - Self-space の Trace / self_state（自己ベクトル）
を受け取り、

    logits' = logits_base + alpha * bias(self_vector)

という形で、自己の幾何を生成分布に反映する。

Level 1:
    - Attention をいじらず、ログitバイアスのみで自己影響を注入
    - それでも「同じプロンプトでも自己状態で出力が変わる」ことは検証可能
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfConditionedGenerator(nn.Module):
    def __init__(
        self,
        base_model,
        tokenizer,
        self_dim: int,
        alpha: float = 0.8,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            base_model: AutoModelForCausalLM 互換モデル (HF)
            tokenizer: 対応する tokenizer
            self_dim: Self-space の次元 (Trace / self_center の次元)
            alpha: Self-conditioning の強さ
        """
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.alpha = alpha

        if device is None:
            device = next(base_model.parameters()).device
        self.device = device

        hidden_dim = base_model.config.hidden_size

        # Self-space -> LM hidden 空間への射影
        self.self_to_hidden = nn.Linear(self_dim, hidden_dim)

        # 語彙埋め込み行列を参照
        self.embed_weight = self.model.get_input_embeddings().weight  # (V, hidden_dim)

    def compute_self_bias(self, self_vec: torch.Tensor) -> torch.Tensor:
        """
        Self ベクトルから vocab 次元のログitバイアスを計算。

        Args:
            self_vec: (self_dim,)

        Returns:
            bias: (vocab_size,)
        """
        self_vec = self_vec.to(self.device)

        # 1) Self 空間 → LLM hidden 空間
        h_self = self.self_to_hidden(self_vec)  # (hidden_dim,)

        # 正規化（方向情報だけ使う）
        h_self = F.normalize(h_self, dim=-1)

        # 2) 語彙埋め込み行列との内積で「自己との整合度」を計算
        #    bias_i = <E_i, h_self>
        bias = torch.matmul(self.embed_weight, h_self)  # (vocab_size,)

        # スケール正規化（オプション）
        bias = bias / (bias.std() + 1e-6)

        return bias  # ここに alpha を掛けるのは呼び出し側

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        self_vec: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """
        Self-conditioned generation を行う簡易サンプラー。

        Args:
            prompt: 入力プロンプト（自然言語）
            self_vec: Self-space ベクトル (self_dim,)
            max_new_tokens: 生成トークン数
            temperature: 温度サンプリング
            top_k: Top-k フィルタリング

        Returns:
            生成テキスト
        """
        self.model.eval()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        bias = self.compute_self_bias(self_vec)  # (V,)
        bias = self.alpha * bias  # Self-conditioning の強さ

        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = self.model(generated)
            logits = outputs.logits[:, -1, :]  # (1, V)

            # Self-conditioned bias を加算
            logits = logits + bias.unsqueeze(0)

            # 温度スケーリング
            logits = logits / max(temperature, 1e-6)

            # Top-k フィルタ
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, indices, values)
                logits = mask

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

            generated = torch.cat([generated, next_token], dim=1)

            # EOS で打ち切りたいならここで判定
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
