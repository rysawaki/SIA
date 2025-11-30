# ============================
# file: trace_extractor.py
# Extract Trace, Shock, Affect from input experience
# ============================

import torch
import torch.nn.functional as F


class TraceExtractor:
    """
    SIAの概念に基づく Trace / Shock / Affect 抽出器

    【原理】
        trace  =  meaning-direction( input - reconstruction )
        shock  =  予測との差異の大きさ (semantic surprise)
        affect =  心的インパクト (self-relevance / valence)

    実装方針:
        - 生成LLMの埋め込み層 or Encoder出力を再利用
        - affect は仮のHeuristic（後に学習型に置換可能）
    """

    def __init__(self, device="cpu"):
        self.device = device

    @torch.no_grad()
    def extract(
        self,
        input_emb: torch.Tensor,   # (d,)  入力文の埋め込み
        recon_emb: torch.Tensor,   # (d,)  モデルの解釈・再構成
        self_state: torch.Tensor,  # (d,)  現在のSelf中心
    ):
        # ===== 1. Trace (semantic difference) =====
        diff = input_emb - recon_emb
        trace = F.normalize(diff, dim=-1)

        # ===== 2. Shock (surprise) =====
        shock_raw = diff.norm().item()  # float
        print(f"[DEBUG] raw shock = {shock_raw:.4f}")
        shock = float(min(max(shock_raw, 0.0), 1.0))  # clamp to [0,1]

        # ===== 3. Affect (self relevance) =====
        cos_sim = torch.dot(
            F.normalize(input_emb, dim=-1),
            F.normalize(self_state, dim=-1)
        )
        affect = float(torch.sigmoid(cos_sim * 5))  # [-1,1] → [0,1]

        return trace, shock, affect
