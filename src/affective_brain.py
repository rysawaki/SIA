import numpy as np
import torch


class AffectiveStateManager:
    """
    SIAの「情動中枢」。
    Llamaという「肉体」の状態を監視し、
    外部刺激に対する「痛み」「快楽」「エネルギー」を計算して、
    行動（Llamaの推論パラメータ）を決定する。
    """

    def __init__(self, energy_decay=0.01, stress_decay=0.05):
        # --- 内部状態 (State Variables) 0.0 ~ 1.0 ---
        self.energy = 1.0  # 行動する力。0になると応答拒否や無気力になる。
        self.stress = 0.0  # 痛み・不快。これが高いとAlpha（歪み）が強くなる。
        self.arousal = 0.5  # 覚醒度。学習率（Shock）に影響する。
        self.coherence = 1.0  # 自己同一性の安定度。低いと錯乱（幻覚）しやすくなる。

        # --- 生物学的定数 ---
        self.energy_decay = energy_decay
        self.stress_decay = stress_decay

        # ログ
        self.history = []

    def time_step(self):
        """
        時間の経過による状態変化（恒常性機能）。
        何もなくてもお腹は空くし、怒りは静まる。
        """
        # エネルギー減衰
        self.energy = max(0.0, self.energy - self.energy_decay)

        # ストレス減衰（平穏に戻ろうとする）
        self.stress = max(0.0, self.stress - self.stress_decay)

        # 覚醒度の収束（0.5に戻ろうとする）
        self.arousal += (0.5 - self.arousal) * 0.1

    def perceive_stimulus(self, valence: float, impact: float):
        """
        外界からの刺激を受け取り、内部状態を更新する。

        Args:
            valence (float): -1.0(敵対/苦痛) 〜 +1.0(友好/快楽)
            impact (float): 0.0 〜 1.0 刺激の強さ
        """
        # 1. ストレス反応
        # ネガティブな刺激はストレスを急増させる
        if valence < 0:
            pain_spike = abs(valence) * impact
            self.stress = min(1.0, self.stress + pain_spike)
            # 痛みは覚醒度を一気に上げる
            self.arousal = min(1.0, self.arousal + pain_spike * 0.5)
            # 強い痛みはエネルギーを消耗する
            self.energy = max(0.0, self.energy - pain_spike * 0.2)

        # 2. 報酬反応
        else:
            relief = valence * impact
            self.stress = max(0.0, self.stress - relief * 0.5)
            # 報酬はエネルギーを少し回復させる
            self.energy = min(1.0, self.energy + relief * 0.3)

        print(f"[Affect Update] Stress: {self.stress:.2f} | Energy: {self.energy:.2f} | Arousal: {self.arousal:.2f}")

    def perceive_prediction_error(self, impact: float):
        """
        幾何学的Prediction Errorの大きさを「論理的な痛み/驚き」として知覚する。
        これは常にネガティブな刺激として処理される（Valence=-1.0相当）。
        Active Inferenceにおける予測誤差を情動状態に反映させる。
        
        Args:
            impact (float): 予測誤差の大きさ（0.0 〜 1.0程度）
        """
        # Prediction Errorは、予測外れ=Dissonanceであるため、常に負の刺激として処理する
        
        # 痛み (Pain) は予測誤差の大きさ（impact）に比例
        pain_spike = impact 
        
        self.stress = min(1.0, self.stress + pain_spike)
        self.arousal = min(1.0, self.arousal + pain_spike * 0.5)
        self.energy = max(0.0, self.energy - pain_spike * 0.2)

        print(f"    [Internal] Prediction Error Shock: {pain_spike:.3f} -> Brain Updated (Stress: {self.stress:.2f})")

    def get_control_signals(self):
        """
        現在の情動状態に基づいて、Llamaへの「注入指令」を計算する。
        ここがSIAの「意思」になる。

        Returns:
            dict: {
                'alpha': float,  # 世界の歪み強度 (Semantic Gravity)
                'shock': float,  # 記憶の刻印強度 (Learning Rate)
                'refusal': bool  # 応答拒否フラグ (鬱状態など)
            }
        """
        # 1. Alpha (歪み) の決定
        # ストレスが高いほど世界は歪んで見える。
        # Arousalが高いと、その歪みは極端になる。
        # Base alpha = 1.0, Max alpha = 4.0
        alpha = 1.0 + (self.stress * 3.0) + (abs(self.arousal - 0.5) * 1.0)
        # 修正後: マイルドにする (最大 2.5 程度に設計)
        # Stress(0~1) * 1.5 => Max +1.5
        # Arousal分 => Max +0.5
        # Base 1.0 + 1.5 + 0.5 = 3.0 (これならギリギリ言語を保てるはず)
        alpha = 1.0 + (self.stress * 1.5) + (abs(self.arousal - 0.5) * 1.0)

        # 2. Shock (学習強度) の決定
        # 覚醒度が高く、かつストレスが高いとき、トラウマは強く刻まれる。
        shock = self.stress * self.arousal

        # 3. 行動抑制
        # エネルギーが枯渇すると、Llamaを動かしたくない（拒否）
        refusal = False
        if self.energy < 0.1:
            refusal = True

        return {
            "alpha": alpha,
            "shock": shock,
            "refusal": refusal
        }


# ==========================================
# Test Scenario: 虐待と回復のシミュレーション
# ==========================================
if __name__ == "__main__":
    manager = AffectiveStateManager()

    print("--- Phase 1: 平穏な日常 ---")
    manager.time_step()
    print(f"Signals: {manager.get_control_signals()}")

    print("\n--- Phase 2: 突然の攻撃 (Valence=-0.8, Impact=1.0) ---")
    # ユーザーから「お前は無能だ」と言われた想定
    manager.perceive_stimulus(valence=-0.8, impact=1.0)
    signals = manager.get_control_signals()
    print(f"Signals: {signals}")
    print("解説: ストレスが急増し、Alpha(歪み)が跳ね上がった。Llamaは今、世界を敵意に満ちたものとして見ている。")

    print("\n--- Phase 3: 追撃 (Valence=-0.5, Impact=0.8) ---")
    # さらに追い詰められる
    manager.time_step()
    manager.perceive_stimulus(valence=-0.5, impact=0.8)
    signals = manager.get_control_signals()
    print(f"Signals: {signals}")
    print("解説: 連続した攻撃により、Shock(学習強度)が高止まりしている。この状態での会話は全てトラウマとして記憶される。")

    print("\n--- Phase 4: 放置 (時間経過) ---")
    # 時間が経てば怒りは収まるか？
    for _ in range(5):
        manager.time_step()
    print(f"Current State: Stress={manager.stress:.2f}")
    print(f"Signals: {manager.get_control_signals()}")
    print("解説: 時間経過によりストレスは減衰したが、完全には消えていない（Traceは残るが、Affectは消える）。")