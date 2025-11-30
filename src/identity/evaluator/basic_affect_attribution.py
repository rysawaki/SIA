# src/identity/evaluator/basic_affect_attribution.py
# -*- coding: utf-8 -*-
"""
Basic Affect & Attribution Estimation (Level 1 Prototype)

SIAパイプラインの第1段階として使用する最小実装。
- Affect（Valence / Arousal）はVADERによる感情スコアを利用
- Attribution（自己関連度）はキーワードベースの簡易推定
  → Level 2 以降のLLMベース判定に置き換え可能な設計

責務:
    ✔ テキストを解析して (valence, arousal, attribution) を返す
    ✘ 幾何学変形・Trace更新・Self-space操作はしない
"""

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER (英語ベースのため日本語入力は簡易的な対応に留まる)
vader = SentimentIntensityAnalyzer()

# Self-related keywords (Level 1用の最小構成。後からLLM抽出に切替可能)
SELF_KEYWORDS = [
    "I", "me", "my", "myself",
    "私", "自分", "僕", "俺", "心", "感じた", "思った", "傷ついた", "孤独", "嬉しい"
]


# ======================================================
# 1. Affect 推定（Valence, Arousal）
# ======================================================
def estimate_affect(text: str) -> tuple[float, float]:
    """
    Affect (Valence, Arousal) の推定

    Returns:
        valence ∈ [-1, 1]   快–不快
        arousal ∈ [0, 1]    感情的強度（絶対値として扱う）

    Level 1:
        - VADERによるcompoundスコアを使用
        - arousalは valenceの絶対値を再スケールして使う
    """
    scores = vader.polarity_scores(text)
    valence = float(scores["compound"])  # [-1,1]

    # Arousalは「どれだけ心が動いたか」
    arousal = float(min(1.0, abs(valence) * 1.4))  # 少し強調倍率

    return valence, arousal


# ======================================================
# 2. Attribution 推定（自己関連性）
# ======================================================
def estimate_attribution(text: str) -> float:
    """
    Attribution（自己関連度）の簡易推定

    Returns:
        attribution ∈ [0, 1]
            0.2 : ほぼ自分と無関係
            0.6 : ある程度自分と関連
            1.0 : 強く自分と関係する／内的経験

    Level 1:
        - keywordsベースの単純判定
        - 後で LLMベース / Context-aware 推定と置換可能
    """
    text_lower = text.lower()
    count = sum(1 for w in SELF_KEYWORDS if w.lower() in text_lower)

    if count == 0:
        return 0.2
    elif count == 1:
        return 0.6
    else:
        return 1.0


# ======================================================
# 3. 全体まとめ（使いやすいユーティリティ）
# ======================================================
def evaluate_affect_and_attribution(text: str) -> dict:
    """
    両方の推定をまとめて実行するユーティリティ関数。

    Returns:
        {
            "attribution": float,
            "valence": float,
            "arousal": float,
        }
    """
    valence, arousal = estimate_affect(text)
    attribution = estimate_attribution(text)

    return {
        "attribution": attribution,
        "valence": valence,
        "arousal": arousal,
    }


# ======================================================
# 4. 動作確認 (単体テスト)
# ======================================================
if __name__ == "__main__":
    samples = [
        "I felt something change in me when I saw that picture.",
        "The mountains are covered in snow.",
        "My heart was quietly broken that day.",
        "This is just a random neutral sentence.",
    ]

    for s in samples:
        print(f"\nText: {s}")
        print(evaluate_affect_and_attribution(s))
