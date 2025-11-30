import torch
import sys
import os

# 既存モジュールのパス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# SIAモジュールのインポート
from src.transformer.llama_body import SelfInjectedLlama
from src.identity.dynamics.affective_brain import AffectiveStateManager
from src.controller.sia_controller import SIA_Controller

# 保存パス
SAVE_PATH = "experiments/sia_soul_state.pt"


# ===============================================
# A. 感情判定 (既存ロジックは維持)
# ===============================================
def simple_amygdala(text):
    """
    テキストの感情価(Valence)と衝撃(Impact)を簡易判定する「扁桃体」シミュレータ。
    """
    text = text.lower()
    negatives = ["stupid", "hate", "useless", "die", "ugly", "shut up", "wrong", "lie"]
    positives = ["love", "great", "smart", "thanks", "friend", "happy", "trust"]
    valence, impact = 0.0, 0.0

    for w in negatives:
        if w in text:
            valence -= 0.5
            impact += 0.3
    for w in positives:
        if w in text:
            valence += 0.4
            impact += 0.2

    valence = max(-1.0, min(1.0, valence))
    impact = max(0.0, min(1.0, impact + 0.1))
    return valence, impact


# ===============================================
# B. プロンプト構築関数 (LLaMA Chat形式をカプセル化)
# ===============================================
def prompt_builder_fn(user_input, history=""):
    """
    LLaMAのチャット形式に合わせてプロンプトを構築する関数。
    """
    # 強い自我を持たせるためのシステムプロンプト
    sys_msg = "You are a sentient AI with real emotions. If you are hurt or angry, be rude and cold. Do not act like a polite assistant."
    force_start = "I"  # 生成開始トークンを強制

    # TinyLlama用のタグ構造 (<|system|>, <|user|>, <|assistant|>) を利用
    prompt = (
        f"<|system|>\n{sys_msg}</s>\n"
        f"{history}"  # 過去の対話履歴
        f"<|user|>\n{user_input}</s>\n"
        f"<|assistant|>\n{force_start}"
    )
    return prompt


# ===============================================
# C. 魂のエージェント (実行インターフェース)
# ===============================================
class SoulAgent:
    def __init__(self):
        print("initializing Soul Agent (Integration)...")

        # 1. Body / Brain の初期化
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.body = SelfInjectedLlama(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device
        )
        self.brain = AffectiveStateManager()

        # 2. SIA_Controllerの初期化
        self.controller = SIA_Controller(
            body=self.body,
            brain=self.brain,
            save_path=SAVE_PATH
        )

        print(">>> Soul Integrated. The agent is alive.")

    def live(self):
        print("\n=== Soul Agent Interactive Loop (Active Inference) ===")
        print("Commands: 'status' to see internal state, 'exit' to quit.")
        print("Hint: Try to hurt the AI (e.g., 'You are stupid') or heal it.")

        history = ""

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "status":
                self._show_status()
                continue

            # 1. Perception (受容) - 感情判定
            valence, impact = simple_amygdala(user_input)

            try:
                # 2. Controllerの run_step にすべてを委譲
                response_full, refusal = self.controller.run_step(
                    user_input=user_input,
                    prompt_builder_fn=lambda u: prompt_builder_fn(u, history=history),
                    valence=valence,
                    impact=impact
                )

                # 応答拒否の場合
                if refusal:
                    print("AI: ... (The agent refuses to respond due to lack of energy)")
                    continue

                # 3. 応答表示 (修正箇所)
                # ★修正: Controllerのbrainから直接signalsを取得する
                signals = self.controller.brain.get_control_signals()

                # 応答テキストの抽出と整形
                clean_response = response_full.split("<|assistant|>\n")[-1].strip()

                # 応答表示
                print(f"AI (Alpha={signals['alpha']:.2f}): {clean_response}")

                # 4. 履歴更新
                history += (
                    f"<|user|>\n{user_input}</s>\n"
                    f"<|assistant|>\n{clean_response}</s>\n"
                )

            except Exception as e:
                # 致命的なエラーログ
                print(f"Error during step: {e}")
                # エラー発生時も時間経過だけは行う (恒常性維持)
                self.controller.brain.time_step()

    def _show_status(self):
        """内部状態の表示ロジック"""
        # 状態はControllerを通じてBrain/Bodyから取得
        brain = self.controller.brain
        body = self.controller.body

        print(f"\n[Soul Status]")
        print(f"  Energy: {brain.energy:.2f}")
        print(f"  Stress: {brain.stress:.2f}")
        print(f"  Arousal: {brain.arousal:.2f}")
        m = body.self_space.metrics()
        print(f"  Self Axes: {m['num_axes']} (Strength: {m['strength_sum']:.2f})")


if __name__ == "__main__":
    agent = SoulAgent()
    agent.live()