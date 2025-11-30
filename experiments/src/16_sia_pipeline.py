# src/experiment/sia_pipeline.py

from identity.encoder.self_aware_encoder import SelfAwareEncoder
from src.identity.engine.imprint_engine import ImprintGeometryEngine, ImprintEvent


def process_input(text, encoder, engine, self_center):
    """
    text: 入力（ユーザーの発話、体験記述、芸術の説明など）
    encoder: SelfAwareEncoder
    engine: ImprintGeometryEngine
    self_center: 現在の self-space の中心

    戻り値:
    - updated_info: 幾何変形後の状態（Trace, Potential, Curvature）
    - u_t: 入力がSelf-space上でどこに写ったか
    """

    # === 1) LLaMA → Self-space embedding (u_t) ===
    u_t = encoder(text, self_center=self_center)

    # === 2) Attribution & Affect の評価 ===
    attribution = estimate_attribution(text)  # ⟵ Dummy, 後でモデル化
    valence, arousal = estimate_affect(text)  # ⟵ Dummy, 後でVADERやLLM評価

    # === 3) ImprintEvent の生成 ===
    event = ImprintEvent(
        u_t=u_t,
        attribution=attribution,
        valence=valence,
        arousal=arousal,
        meta={"source_text": text}
    )

    # === 4) 幾何更新 ===
    info = engine.update_from_event(
        event,
        update_curvature=True,
        k_for_curvature=16
    )

    # === 5) 新たな自己中心の更新 ===
    engine.set_self_center(info["trace_vec"])  # Traceを自己中心にする例

    return info, u_t
