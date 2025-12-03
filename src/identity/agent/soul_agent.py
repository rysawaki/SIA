import torch

# SIAモジュールのインポート
from src.transformer.llama_body import SelfInjectedLlama
from src.identity.dynamics.affective_brain import AffectiveStateManager
from src.controller.sia_controller import SIAController
from utils.prompt_builder import prompt_builder
from identity.dynamics.affect_estimators import simple_amygdala
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
        self.controller = SIAController(
            body=self.body,
            growth_kernel=self.brain.growth_kernel if hasattr(self.brain, 'growth_kernel') else None,
            trace_tensor=self.brain.trace_tensor if hasattr(self.brain, 'trace_tensor') else None,
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
                response_text, signals = self.run_step(
                    user_input=user_input,
                    prompt_builder_fn=lambda u: prompt_builder(u, history),
                    valence=valence,
                    impact=impact
                )

                # 応答拒否の場合
                if signals:
                    print("AI: ... (The agent refuses to respond due to lack of energy)")
                    continue

                # 3. 応答表示 (修正箇所)
                # ★修正: Controllerのbrainから直接signalsを取得する
                signals = self.controller.brain.get_control_signals()

                # 応答テキストの抽出と整形
                clean_response = response_text.split("<|assistant|>\n")[-1].strip()

                # 応答表示
                print(f"AI (α={signals['alpha']:.2f}, Stress={signals['stress']:.2f}): {response_text}")
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

    def run_step(self, user_input, prompt_builder_fn, valence, impact):
        """
        Agentの1ステップ処理:
        - Brainで応答生成
        - ControllerでIdentity更新（SelfSpace / Trace / Affect）
        - 次の状態をBrainに反映
        """
        # 1️⃣ プロンプト生成
        prompt = prompt_builder_fn(user_input)

        # 2️⃣ Brainによる応答生成
        response_text = self.body.generate(prompt)

        # 3️⃣ ControllerでIdentityを更新
        self.controller.update_identity(
            input_text=user_input,
            output_text=response_text,
            valence=valence,
            impact=impact
        )

        # 4️⃣ Brainの状態も更新（arousal, stressなど）
        if hasattr(self.brain, "update_state"):
            self.brain.update_state(valence, impact)

        signals = {
            "alpha": float(self.controller.body.self_space.alpha.item()),
            "arousal": float(self.brain.arousal),
            "stress": float(self.brain.stress),
        }

        return response_text, signals

    def save_soul(self):
        soul_state = self.controller.collect_soul_state()
        self.soul_manager.save_soul(
            soul_state,
            generation=self.get_generation_id()
        )

    def load_soul(self, path):
        soul_state = self.soul_manager.load_soul(path)
        self.controller.integrate_soul_state(soul_state)
