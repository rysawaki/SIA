# src/affective_brain.py (Public Edition)
class AffectiveStateManager:
    """
    [SIA Core: Affective Interface]
    This is a template for the 'Soul'. 
    The logic for calculating Energy, Stress, and Homeostasis is left to the user.
    """
    def __init__(self):
        self.energy = 1.0
        self.stress = 0.0
        self.arousal = 0.5

    def time_step(self):
        pass

    def perceive_stimulus(self, valence: float, impact: float):
        # 公開版ではログを出すだけ
        print(f"[SIA-Core] Stimulus Received: Valence={valence}, Impact={impact}")

    def get_control_signals(self):
        # 公開版は常に「冷静(Alpha=1.0)」で「学習しない(Shock=0.0)」状態を返す
        return {
            "alpha": 1.0,
            "shock": 0.0,
            "refusal": False
        }