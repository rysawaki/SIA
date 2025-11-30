# src/identity/evaluator/basic_affect_attribution.py
# -*- coding: utf-8 -*-
"""
Affect & Attribution Estimation (Level 2 Prototype)

SIAに適合する再設計:
    - Affect = Self-spaceを変形させる力（幾何エネルギー）
    - Attribution = Self軸との整合性（Identity一致度）
    - VADERや辞書型スコアは使用しない

アーキテクチャ:
    ◼ LLM-based semantic scoring (静的)
    ◼ 後で Self-aware 評価に置き換え可能な構造
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AffectAttributionEvaluator:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def estimate(self, text: str) -> dict:
        """
        LLMに直接質問し、JSON形式で意味評価を得る。
        解析観点:
            - valence: 「これは自分を前に進めるか / 傷つけるか」
            - arousal: 「どれだけ気持ちを揺さぶるか」
            - attribution: 「これは私に関係あるか？」
        """
        prompt = f"""
        You are an evaluator of meaningful experience.
        Analyze the following text and respond in valid JSON (ONLY):

        Text: "{text}"

        Evaluate:
        1. valence: (-1 to +1) negative vs. positive existential impact  
        2. arousal: (0 to 1) emotional activation level (how deeply it moves identity)  
        3. attribution: (0 to 1) personal relevance / identity shaping potential  

        Example format:
        {{"valence": 0.75, "arousal": 0.62, "attribution": 0.85}}
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=180)

        return self._parse_output(outputs)

    def _parse_output(self, outputs):
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            import json
            json_str = text[text.find("{") : text.rfind("}") + 1]
            result = json.loads(json_str)

            # Safety clamp
            return {
                "valence": float(max(-1, min(1, result.get("valence", 0)))),
                "arousal": float(max(0, min(1, result.get("arousal", 0)))),
                "attribution": float(max(0, min(1, result.get("attribution", 0)))),
            }
        except:
            # Safe defaults
            return {"valence": 0.0, "arousal": 0.3, "attribution": 0.2}


# ================================
# 単体テスト
# ================================
if __name__ == "__main__":
    evaluator = AffectAttributionEvaluator()
    samples = [
        "I still remember the night when I felt completely seen for the first time.",
        "The desert looks dry and empty.",
        "There is a well in the middle of the desert.",
    ]

    for s in samples:
        print(f"\nText: {s}")
        print(evaluator.estimate(s))
