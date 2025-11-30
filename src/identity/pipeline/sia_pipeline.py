# src/identity/pipeline/sia_pipeline.py
# -*- coding: utf-8 -*-

"""
SIA Processing Pipeline

入力（テキスト）を処理し、
    1) SelfAwareEncoder で u_t を得る
    2) Affect / Attribution を推定
    3) ImprintEvent を生成
    4) ImprintGeometryEngine に渡して自己幾何を更新
    5) Self-center を更新（自己の重心の移動）
    6) 変形後の Self-space 状態 (Trace, Potential, Curvature) を返す

このモジュールは「統制層」
    → 幾何変形や感情推定の中身は知らない
    → モジュール間の接続のみ担当
"""

from identity.encoder.self_aware_encoder import SelfAwareEncoder
from identity.engine.imprint_engine import ImprintEvent, ImprintGeometryEngine
from identity.evaluator.basic_affect_attribution import (
    estimate_affect,
    estimate_attribution,
)

import torch


def process_input(text: str, encoder: SelfAwareEncoder, engine: ImprintGeometryEngine):
    """
    入力を受け取ってSIAの自己変形ループを1ステップ実行する。
    実行のたびに Self-space や Trace が変化し、
    次の入力での反応にも影響する。

    Args:
        text: 入力文章（自然言語）
        encoder: SelfAwareEncoder インスタンス
        engine: ImprintGeometryEngine インスタンス

    Returns:
        info: Self-space 幾何変形結果
        u_t: その入力が自己空間のどこに写ったか（座標）
    """

    # === 1) LLaMA → Self-space 座標埋め込み ===
    self_center = engine.self_center.clone().detach()
    u_t = encoder(text, self_center=self_center)

    # === 2) Attribution & Affect の推定 ===
    attribution = estimate_attribution(text)
    valence, arousal = estimate_affect(text)

    # === 3) ImprintEvent を生成 ===
    event = ImprintEvent(
        u_t=u_t,
        attribution=attribution,
        valence=valence,
        arousal=arousal,
        meta={"source_text": text},
    )

    # === 4) ImprintEngine に渡して幾何更新する ===
    info = engine.update_from_event(
        event,
        update_curvature=True,
        k_for_curvature=16,
    )

    # === 5) Self-center を Trace に更新（Identity重心の移動）===
    engine.set_self_center(info["trace_vec"])

    return info, u_t


# ======================================================
# 📌 簡易デモ（自己変形ループの動作確認用）
# ======================================================
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LLaMAロード（HuggingFaceの仮例）
    llama = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M",  # 仮モデル（LLaMAではない）
        output_hidden_states=True
    ).to(device)
    hidden_dim = llama.config.hidden_size

    # Self-space & Trace & Engine 準備
    from src.identity.engine.imprint_engine import SelfSpace, TraceTensor, ImprintGeometryEngine

    latent_dim = 64
    num_points = 256
    self_space = SelfSpace(latent_dim=latent_dim, num_points=num_points, device=device)
    trace = TraceTensor(latent_dim=latent_dim, device=device)
    engine = ImprintGeometryEngine(self_space, trace, alpha_metric=1.2).to(device)

    # Encoder
    encoder = SelfAwareEncoder(llama, hidden_dim=hidden_dim, self_dim=latent_dim).to(device)

    # === デモ入力 ===
    sentences = [
        "The desert is silent, but I felt something alive within it.",
        "The sky is blue. There is nothing special happening.",
        "I remember when I was lonely, and the world felt distant.",
    ]

    for text in sentences:
        print(f"\n📝 Input: {text}")
        info, u_t = process_input(text, encoder, engine)

        print("Trace Vec Norm:", info["trace_vec"].norm().item())
        print("Mean Potential:", info["potential"].mean().item())
        print("Curvature Magnitude:", info["curvature"].abs().mean().item())
