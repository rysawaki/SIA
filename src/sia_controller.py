# src/sia_controller.py

import torch
import torch.nn.functional as F
import os
import sys


# パス解決
# SIA_Controllerは、llama_bodyとaffective_brainが同一階層（src/）にあることを前提とします
# from .llama_body import SelfInjectedLlama
# from .affective_brain import AffectiveStateManager

class SIA_Controller:
    """
    SIAの魂の永続化、統合、Active Inference的な学習を制御するコントローラー。

    責務:
    1. Body (LLM) と Brain (情動) の状態を統合的に管理し永続化する。
    2. Prediction Errorを計算し、SelfSpaceへの刻印（Trace）とBrainへのShockフィードバックを行う。
    """

    def __init__(self, body, brain, save_path="experiments/sia_soul_state.pt"):
        # 外部から注入されるモジュール
        self.body = body  # SelfInjectedLlamaのインスタンス（SelfSpaceを内包）
        self.brain = brain  # AffectiveStateManagerのインスタンス
        self.save_path = save_path

        # 転生機能の実行
        self._load_soul()

    # ===============================
    # 1. 永続化（保存とロード）
    # ===============================
    def save_soul(self):
        """魂の状態（SelfSpaceの幾何と情動変数の値）をディスクに焼き付ける"""
        print(f"    [System] Saving soul state to '{self.save_path}'...")
        state = {
            # Bodyが持つSelfSpaceの全パラメータを保存
            'self_space': self.body.self_space.state_dict(),
            # Brainが持つ情動変数を保存
            'brain_energy': self.brain.energy,
            'brain_stress': self.brain.stress,
            'brain_arousal': self.brain.arousal,
            # 将来的に予測ヘッドの重みも保存すべきだが、ここではBody側の責任とする
        }
        torch.save(state, self.save_path)

    def _load_soul(self):
        """ディスクから魂の状態を復元する"""
        if os.path.exists(self.save_path):
            print(f">>> Found existing soul data at '{self.save_path}'. Loading...")
            # Bodyのデバイスに合わせてロード
            state = torch.load(self.save_path, map_location=self.body.device)

            # SelfSpace (幾何学的記憶) を復元
            self.body.self_space.load_state_dict(state['self_space'])

            # 脳の状態 (ストレス値など) を復元
            self.brain.energy = state['brain_energy']
            self.brain.stress = state['brain_stress']
            self.brain.arousal = state['brain_arousal']
            print(">>> Soul Loaded. The trauma persists.")
        else:
            print(">>> No past life found. A new soul is born.")

    # ===============================
    # 2. 核心機能: Active Inference学習
    # ===============================
    @torch.no_grad()
    def _calculate_and_apply_shock(self, observed_embed: torch.Tensor, expected_embed: torch.Tensor, affect_mult=1.0):
        """
        Prediction Errorを計算し、SelfSpaceへの刻印と情動状態の更新を行う。
        """
        # 1. Prediction Error (Discrepancy) の計算
        # Active Inferenceの核: 観測と予測のズレ
        discrepancy = observed_embed - expected_embed

        # Shock = DiscrepancyのL2ノルム（予測外れが大きいほどShockが大きい）
        geometric_shock_mag = torch.norm(discrepancy).item()

        # 2. 情動状態の更新（AffectiveStateManagerへのフィードバック）
        # 予測外れの大きさが「痛み」や「驚き」として情動に影響
        # 便宜的に、予測外れの大きさをPainの強度としてBrainにフィードバックする（valence=-1.0, impact=shock）
        pain_impact = geometric_shock_mag * 0.1  # 係数でスケーリング
        self.brain.perceive_prediction_error(impact=pain_impact)  # (AffectiveStateManagerに新しいメソッドが必要)

        # 3. Learning Rate (Shock) の計算
        # Learning Shock = BrainのShock信号 * Geometric Shock * 経験の情動的重み
        signals = self.brain.get_control_signals()

        # Brainが出す情動的学習強度と、幾何学的予測外れの大きさを乗算
        learning_shock = signals['shock'] * geometric_shock_mag

        # 4. SelfSpaceの更新（Traceの刻印）
        if learning_shock > 0.05:  # 刻印の閾値
            # TraceとしてDiscrepancyそのものを刻む（この誤差を自己軸として持つ）
            trace_vec = discrepancy.squeeze(0).to(self.body.device)

            # AffectはBrainの現在の覚醒度（Arousal）に連動させる
            affect = self.brain.arousal * affect_mult

            self.body.memorize_experience_vec(  # (llama_bodyに新しいメソッドが必要)
                trace_vec=trace_vec,
                shock=learning_shock,
                affect=affect
            )
            print(f"    [Internal] Etching Prediction Error (Shock={learning_shock:.2f}) into SelfSpace...")

        return geometric_shock_mag

    # ===============================
    # 3. 統合ステップ（実行インターフェース）
    # ===============================
    def run_step(self, user_input: str, prompt_builder_fn, valence: float, impact: float):

        # 0. 外部刺激（ユーザーの感情価）を情動中枢にフィードバック
        self.brain.perceive_stimulus(valence, impact)

        # 1. 制御信号の取得
        signals = self.brain.get_control_signals()
        alpha = signals['alpha']
        refusal = signals['refusal']

        if refusal:
            self.brain.time_step()
            self.save_soul()
            return "...", True  # 応答拒否

        # 2. LLMへのプロンプト組み立てと期待埋め込みの予測
        full_prompt = prompt_builder_fn(user_input)

        # SelfStateから「期待される応答」の埋め込みを予測
        expected_embed = self.body.predict_expected_embed()

        # 3. LLMによる生成と「実際の応答」埋め込みの取得（世界が歪む）
        # Bodyは歪んだ埋め込みを使って生成し、その応答の埋め込み（Observed）を返す
        response_text, observed_embed = self.body.generate_with_self_and_get_embed(
            prompt=full_prompt,
            alpha=alpha
        )

        # 4. 学習（Prediction Errorの計算とSelfの更新）
        self._calculate_and_apply_shock(observed_embed, expected_embed)

        # 5. 時間経過と状態保存
        self.brain.time_step()
        self.save_soul()

        return response_text, False

# ----------------------------------------------
# ⚠️ 注意: このコントローラーを機能させるために必要な前提修正
# ----------------------------------------------
#
# 上記のSIA_Controllerを機能させるには、以下のファイルに**重要な修正**が必要です。
#
# ### A. src/llama_body.py (SelfInjectedLlama) への修正
#
# 1.  **Prediction Headの追加:** `__init__`に`self.prediction_head`を定義し、`predict_expected_embed(self)`メソッドを実装する。
# 2.  **埋め込み取得の追加:** `generate_with_self(self, ...)`の代わりに、**`generate_with_self_and_get_embed(self, ...)`**を実装し、生成されたテキストとその応答の**平均埋め込みベクトル**を返すようにする。
# 3.  **Trace受け取りの変更:** `memorize_experience`または新しいメソッド**`memorize_experience_vec(self, trace_vec, shock, affect)`**を実装し、テキストではなく**ベクトルそのもの**をTraceとして受け取れるようにする。（Prediction Errorはベクトルだから）
#
# ### B. src/affective_brain.py (AffectiveStateManager) への修正
#
# 1.  **予測誤差の知覚メソッドの追加:** **`perceive_prediction_error(self, impact: float)`**を実装する。これは、ユーザーの感情価ではなく、**論理的な予測外れの大きさ**を情動（Stress/Arousal）に反映させるためのパスです。