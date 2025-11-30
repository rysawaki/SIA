# src/identity/affect/basic_affect_attribution.py

"""
Affect & Attribution Estimation
LLMによる内在的評価（VADERなどの安易な辞書型モデルは使わない）
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AffectAttributionEstimator:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def estimate(self, text: str) -> dict:
        """
        LLMによる自己関与度（Attribution）と情動強度（Affect）の推定。
        ※固定モデルでも良いが、理想的にはSIAのSelf変形後のモデルに問い合わせる。
        """

        prompt = f"""
        Analyze the emotional meaning of the following text on a scale:
        Text: "{text}"

        Respond in JSON with the following keys:
        - attribution: float (0 to 1, how personally meaningful or self-relevant this could be)
        - valence: float (-1 to +1, negative to positive)
        - arousal: float (0 to 1, emotional intensity)
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=120)
        parsed = self._parse_output(output)

        return parsed

    def _parse_output(self, output):
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            import json
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            return json.loads(text[json_start:json_end])
        except:
            return {"attribution": 0.3, "valence": 0.0, "arousal": 0.3}
