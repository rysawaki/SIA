# -*- coding: utf-8 -*-
"""
Affect Estimators:
 - simple_amygdala: keyword-based valence & impact estimation
"""

def simple_amygdala(text: str):
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
