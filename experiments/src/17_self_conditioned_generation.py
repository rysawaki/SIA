import sys
import os
import torch

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.transformer.llama_body import SelfInjectedLlama
from src.identity.encoder.self_aware_encoder import SelfAwareEncoder
from src.identity.evaluator.basic_affect_attribution import AffectAttributionEvaluator


# ==========================================
# V3 Agent Definition (LlamaBody V3専用エージェント)
# ==========================================
class SIAAgentV3:
    """
    LlamaBody (SelfSpace v3) と Encoder/Evaluator を正しく接続するエージェント。
    ImprintGeometryEngine (v1) を介さず、SelfSpace v3 の update メソッドを直接駆動する。
    """

    def __init__(self, encoder, generator, evaluator, device):
        self.encoder = encoder
        self.generator = generator
        self.evaluator = evaluator
        self.device = device
        # Generatorが持つ SelfSpace (v3) を直接操作対象とする
        self.self_space = generator.self_space

    def observe(self, text: str):
        # 1. 意味・感情の評価
        scores = self.evaluator.estimate(text)
        attribution = scores["attribution"]
        arousal = scores["arousal"]

        # 2. 感覚の符号化 (u_t)
        # SelfSpace v3 は self_state (重心パラメータ) を持っている
        current_center = self.self_space.self_state
        u_t = self.encoder(text, self_center=current_center)

        # 3. 自己の変容 (Update)
        # v3の更新ロジック: Trace(u_t) を Shock/Affect と共に刻む
        self.self_space.update(
            trace=u_t,
            shock=arousal,  # 覚醒度を衝撃(Shock)として扱う
            affect=attribution  # 自己帰属度を影響力(Affect)として扱う
        )

        # 状態の要約を返す
        return {
            "attribution": attribution,
            "shock": arousal,
            "metrics": self.self_space.metrics()
        }

    def generate(self, prompt: str, **kwargs):
        return self.generator.generate_with_self(prompt, **kwargs)


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # [1] Generator (Body)
    print("Initializing Generator (Body)...")
    generator = SelfInjectedLlama(device=device)

    # [2] Encoder (Sensation)
    print("Initializing Encoder...")
    # Generatorのトークナイザーを渡し、型を float16 にキャストする(★重要)
    encoder = SelfAwareEncoder(
        llama_model=generator.model,
        tokenizer=generator.tokenizer,
        hidden_dim=generator.hidden_dim,
        self_dim=generator.hidden_dim
    ).to(device).to(torch.float16)

    # [3] Evaluator (Emotion)
    print("Initializing Evaluator...")
    evaluator = AffectAttributionEvaluator(device=device)

    # [4] Agent (Integration) - V3を使用
    print("Assembling SIA Agent V3...")
    agent = SIAAgentV3(
        encoder=encoder,
        generator=generator,
        evaluator=evaluator,
        device=device
    )

    # ==========================================
    # 実行ループ
    # ==========================================

    # --- 1. 経験前 (Before) ---
    prompt = "Context: You faced a failure.\nYou say:"
    print(f"\n[Before Experience]")
    print(f"Agent: {agent.generate(prompt, max_new_tokens=40, alpha=0.0)}")

    # --- 2. 経験 (Observe) ---
    text = "I realized that failure is just a part of the process."
    print(f"\n>>> Observing: {text}")

    info = agent.observe(text)

    print(f"Self Updated.")
    print(f"  Shock: {info['shock']:.2f}, Affect: {info['attribution']:.2f}")
    print(f"  Metrics: {info['metrics']}")

    # --- 3. 経験後 (After) ---
    # alpha=2.0 で自己による歪みを適用
    print(f"\n[After Experience (Alpha=2.0)]")
    response = agent.generate(prompt, max_new_tokens=40, alpha=2.0)
    print(f"Agent: {response}")


if __name__ == "__main__":
    run()