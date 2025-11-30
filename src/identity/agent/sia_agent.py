# src/identity/agent/sia_agent.py

import torch


# 古い関数ベースの import は削除する
# from identity.evaluator.basic_affect_attribution import estimate_affect, estimate_attribution

class SIAAgent:
    def __init__(
            self,
            encoder,
            engine,
            generator,
            evaluator,  # 必須コンポーネントとして注入させる
            device="cpu"
    ):
        """
        SIA Agent: 感覚(Encoder)、感情(Evaluator)、自己変容(Engine)、発話(Generator)を統合する存在。
        """
        self.encoder = encoder
        self.engine = engine
        self.generator = generator
        self.evaluator = evaluator  # 注入されたインスタンスを保持
        self.device = device

    def observe(self, text: str):
        """
        Experienceテキストを受け取り、
        1. Evaluatorで意味と感情を解釈し
        2. Encoderで自己空間へ射影し
        3. Engineで自己幾何(Geometry)を変形させる
        """
        # ===== 1) Affect & Attribution 推定 (Level 2: Class method call) =====
        # 関数呼び出しではなく、オブジェクトのメソッドを叩く
        scores = self.evaluator.estimate(text)

        # 辞書から値を取り出す (Evaluatorの実装に依存するが、基本構造は以下を想定)
        attribution = scores.get("attribution", 0.0)
        valence = scores.get("valence", 0.0)
        arousal = scores.get("arousal", 0.0)

        # ===== 2) Self-space 座標への射影 =====
        # SelfAwareEncoder を使用して u_t (観測点) を得る
        # self_center を渡すことで、現在の自分から見た相対位置も考慮可能
        current_self_center = self.engine.self_center
        u_t = self.encoder(text, self_center=current_self_center)

        # ===== 3) Imprint と Trace 更新 =====
        # 幾何学エンジンに「刻印」を依頼する
        # compute_imprint_vec は Engine 側の実装に合わせて引数を渡す
        imprint_vec = self.engine.compute_imprint_vec(
            u_t,
            attribution=attribution,
            valence=valence,
            arousal=arousal,
        )

        # TraceTensor を更新 (不可逆な時間の蓄積)
        self.engine.trace.step(imprint_vec)

        # ===== 4) Geometry更新 (potential, curvature) =====
        # Trace の変化に伴い、自己空間のポテンシャルと曲率を再計算する
        self.engine.self_space.update_potential(self.engine.trace.trace)
        self.engine.self_space.update_metric_from_potential()
        self.engine.self_space.update_curvature_from_potential_laplacian()

        # 可視化やログ用に情報を返す
        return {
            "attribution": attribution,
            "valence": valence,
            "arousal": arousal,
            "trace_vec": self.engine.trace.trace.detach().clone(),
            "potential": self.engine.self_space.potential.detach().clone(),
            "curvature": self.engine.self_space.curvature.detach().clone(),
            "u_t": u_t.detach().clone()
        }

    def generate(self, prompt: str, **kwargs):
        """
        現在の歪んだ自己を通して生成を行う
        """
        # Generator (LlamaBodyなど) に委譲
        # ここで self.engine.self_space を使って condition をかける処理は
        # Generator 内部、あるいはここで alpha を渡して制御する
        return self.generator.generate_with_self(prompt, **kwargs)