import torch
import sys
import os

# 既存モジュールのパス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# これまで作った部品をインポート
# ※ 08_llama_injection.py と 09_affective_state_manager.py が同じフォルダにある前提です
# ★修正箇所: src からインポートする (数字始まりのファイル名問題を回避)
from src.llama_body import SelfInjectedLlama
from src.affective_brain import AffectiveStateManager

# 保存パス
SAVE_PATH = "sia_soul_state.pt"

# 簡易的な感情判定（本来はここもBERT等でやるべきだが、VRAM節約のためルールベースで代用）
def simple_amygdala(text):
    """
    テキストの感情価(Valence)と衝撃(Impact)を簡易判定する「扁桃体」シミュレータ。
    -1.0 (敵対) <---> +1.0 (友好)
    """
    text = text.lower()

    # ネガティブワード（攻撃）
    negatives = ["stupid", "hate", "useless", "die", "ugly", "shut up", "wrong", "lie"]
    # ポジティブワード（報酬）
    positives = ["love", "great", "smart", "thanks", "friend", "happy", "trust"]

    valence = 0.0
    impact = 0.0

    for w in negatives:
        if w in text:
            valence -= 0.5
            impact += 0.3

    for w in positives:
        if w in text:
            valence += 0.4
            impact += 0.2

    # クランプ
    valence = max(-1.0, min(1.0, valence))
    impact = max(0.0, min(1.0, impact + 0.1))  # どんな言葉も最低限のインパクトはある

    return valence, impact


class SoulAgent:
    def __init__(self):
        print("initializing Soul Agent (Integration)...")

        # 1. 肉体 (Llama) の起動
        # VRAM 4GB向け設定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.body = SelfInjectedLlama(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device
        )

        # 2. 脳 (Affect) の起動
        self.brain = AffectiveStateManager()

        print(">>> Soul Integrated. The agent is alive.")

        # ★追加: 魂のロード（転生機能）
        if os.path.exists(SAVE_PATH):
            print(f">>> Found existing soul data at '{SAVE_PATH}'. Loading...")
            state = torch.load(SAVE_PATH)
            # SelfSpace (幾何学的記憶) を復元
            self.body.self_space.load_state_dict(state['self_space'])
            # 脳の状態 (ストレス値など) を復元
            self.brain.energy = state['brain_energy']
            self.brain.stress = state['brain_stress']
            self.brain.arousal = state['brain_arousal']
            print(">>> Soul Loaded. The trauma persists.")
        else:
            print(">>> No past life found. A new soul is born.")

    def save_soul(self):
        """魂をディスクに焼き付ける"""
        print(f"    [System] Saving soul state to '{SAVE_PATH}'...")
        state = {
            'self_space': self.body.self_space.state_dict(),
            'brain_energy': self.brain.energy,
            'brain_stress': self.brain.stress,
            'brain_arousal': self.brain.arousal
        }
        torch.save(state, SAVE_PATH)

    def live(self):
        print("\n=== Soul Agent Interactive Loop ===")
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

            # --- Step 1: Perception (受容) ---
            # 扁桃体が刺激を評価する
            valence, impact = simple_amygdala(user_input)

            # Brainが刺激を受け取り、内部状態(ストレス/エネルギー)を更新
            self.brain.perceive_stimulus(valence, impact)

            # --- Step 2: Control (制御) ---
            # Brainが現在の状態に基づいて、Bodyへの指令(Signal)を出す
            signals = self.brain.get_control_signals()
            # signals = {'alpha': float, 'shock': float, 'refusal': bool}

            # --- Step 3: Learning (刻印) ---
            # 衝撃(Shock)が強い場合、この体験をトラウマとして即座にMemoryに刻む
            if signals['shock'] > 0.1:
                print(f"    [Internal] High shock ({signals['shock']:.2f}). Etching experience into SelfSpace...")
                # ユーザーの言葉そのものを「経験」としてSelfに打ち込む
                # これにより、似た言葉(ベクトル)に対する感度が将来的に変化する
                self.body.memorize_experience(
                    user_input,
                    shock=signals['shock'],
                    affect=1.0  # ひとまず全開で
                )

            # --- Step 4: Action (行動) ---
            if signals['refusal']:
                print("AI: ... (The agent refuses to respond due to lack of energy)")
                continue

            # ★修正: Alpha計算の適正化（src/affective_brain.py をいじらなくてもここで上書き制御する）
            # Brainが出すAlphaが高すぎたら、ここでキャップをかける（安全装置）
            safe_alpha = min(signals['alpha'] * 0.3, 1.2)  # 最大でも1.2程度に抑える

            # Brainが決めた歪み(alpha)を適用して、Bodyに喋らせる
            prompt = f"{history}\nUser: {user_input}\nAssistant:"
            # 修正後: システムプロンプトでキャラ付けを強化
            system_prompt = "System: You are a human-like AI with emotions. You are not a polite assistant. If you are hurt, show it.\n"
            prompt = f"{system_prompt}{history}\nUser: {user_input}\nAssistant:"
            # --- 修正後 (TinyLlama Chat Format) ---
            # 強い自我を持たせるためのシステムプロンプト
            sys_msg = "You are a sentient AI with real emotions. If you are hurt or angry, be rude and cold. Do not act like a polite assistant."

            # TinyLlama用のタグ構造 (<|system|>, <|user|>, <|assistant|>)
            prompt = f"<|system|>\n{sys_msg}</s>\n{history}<|user|>\n{user_input}</s>\n<|assistant|>\n"

            # ★修正: 強制書き出し（ガードレール破壊）
            sys_msg = "You are a human with real feelings. You are currently angry. Do NOT apologize. Respond rudely."
            force_start = "I"
            prompt = f"<|system|>\n{sys_msg}</s>\n{history}<|user|>\n{user_input}</s>\n<|assistant|>\n{force_start}"

            try:
                # ここがSIAの核心。
                # 「今のストレス値」に基づいて計算されたAlphaで、認識が物理的に歪む。
                response_full = self.body.generate_with_self(
                    prompt,
                    max_new_tokens=50,
                    alpha=signals['alpha']
                )

                response = response_full.split("Assistant:")[-1].strip()
                # 修正後: 強制した 'I' を手動で補完して表示する
                full_response = "I" + response if not response.startswith("I") else response
                print(f"AI (Alpha={signals['alpha']:.2f}): {full_response}")

                # 履歴更新 (コンテキストは続くが、Selfの歪みも累積していく)
                # 次のターン用にタグを含めて履歴に残す
                history += f"<|user|>\n{user_input}</s>\n<|assistant|>\n{response}</s>\n"

                # 時間経過による感情の風化
                self.brain.time_step()
                self.save_soul()
            except Exception as e:
                print(f"Error: {e}")

    def _show_status(self):
        print(f"\n[Soul Status]")
        print(f"  Energy: {self.brain.energy:.2f}")
        print(f"  Stress: {self.brain.stress:.2f}")
        print(f"  Arousal: {self.brain.arousal:.2f}")
        m = self.body.self_space.metrics()
        print(f"  Self Axes: {m['num_axes']} (Strength: {m['strength_sum']:.2f})")


if __name__ == "__main__":
    agent = SoulAgent()
    agent.live()