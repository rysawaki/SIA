import numpy as np
import torch


class AffectiveStateManager:
    """
    [SIA Core: Affective Interface]
    Manages the 'Jōcho' (Aesthetic Intuition) dynamics of the agent.

    Philosophy:
        The agent seeks 'Aesthetic Harmony' (Low Stress) and avoids 'Dissonance' (High Stress).
        It does not maximize external reward, but maintains internal homeostasis.
    """

    def __init__(self, energy_decay=0.01, stress_decay=0.05):
        # --- Internal State (0.0 ~ 1.0) ---
        # [Physics] Vitality / Energy level. Decays over time.
        # [Philosophy] 'Spring Field' energy. When 0, the agent enters 'Refusal' (Depression).
        self.energy = 1.0

        # [Physics] Entropy / Prediction Error accumulation.
        # [Philosophy] 'Dissonance' (Kokoro no Yodomi). High stress distorts perception (Alpha).
        self.stress = 0.0

        # [Physics] Learning Rate modulator / Sensitivity.
        # [Philosophy] 'Resonance' capability.
        self.arousal = 0.5

        # Constants
        self.energy_decay = energy_decay
        self.stress_decay = stress_decay

    def time_step(self):
        """
        [Homeostasis Loop]
        Proceeds internal time. Energy decays, Stress settles.
        """
        # 1. Energy Decay (Entropy increase)
        # Even without action, living costs energy.
        self.energy = max(0.0, self.energy - self.energy_decay)

        # 2. Stress Recovery (Seeking Harmony)
        # The mind tries to return to a calm state naturally.
        self.stress = max(0.0, self.stress - self.stress_decay)

        # 3. Arousal Convergence (Return to neutral)
        self.arousal += (0.5 - self.arousal) * 0.1

    def perceive_stimulus(self, valence: float, impact: float):
        """
        [Jōcho Perception]
        Receives external stimulus and updates internal affective state.

        Args:
            valence: -1.0 (Pain/Dissonance) to +1.0 (Pleasure/Harmony)
            impact:  0.0 to 1.0 (Intensity of the experience)
        """
        # [Physics] State update logic
        if valence < 0:
            # Pain increases Stress (Dissonance)
            pain = abs(valence) * impact
            self.stress = min(1.0, self.stress + pain)

            # Pain spikes Arousal (Shock)
            self.arousal = min(1.0, self.arousal + pain * 0.5)

            # Pain consumes Energy (Defense mechanism cost)
            self.energy = max(0.0, self.energy - pain * 0.2)

            print(f"[Affect] Dissonance detected. Stress: {self.stress:.2f}")
        else:
            # Harmony reduces Stress
            relief = valence * impact
            self.stress = max(0.0, self.stress - relief * 0.5)

            # Harmony restores Energy (Healing)
            self.energy = min(1.0, self.energy + relief * 0.3)

            print(f"[Affect] Harmony felt. Energy: {self.energy:.2f}")

    def get_control_signals(self):
        """
        [Somatic Output]
        Calculates the distortion parameters for the LLM body based on current affect.
        """
        # 1. Alpha (Geometric Distortion)
        # High stress = High distortion (The world looks hostile)
        # alpha = 1.0 (Normal) ~ 5.0 (Hallucination/Defense)
        alpha = 1.0 + (self.stress * 4.0)

        # 2. Shock (Imprint Strength)
        # High Arousal + High Stress = Traumatic Imprint
        # Only meaningful experiences are carved into memory.
        shock = self.stress * self.arousal

        # 3. Refusal (Action Inhibition)
        # No energy = No action (Depressive state)
        refusal = False
        if self.energy < 0.1:
            refusal = True

        return {
            "alpha": alpha,
            "shock": shock,
            "refusal": refusal
        }