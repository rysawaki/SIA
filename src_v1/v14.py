import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ===========================================================
# 1. Config: 構造を支えるパラメータ（ただのゲイン調整ではなく、方程式の係数）
# ===========================================================

@dataclass
class SIAConfig1D:
    # 環境
    x_min: float = 0.0
    x_max: float = 1.0
    goal_pos: float = 0.8
    max_steps: int = 200

    # Trace の形
    trace_channels: int = 3
    trace_depth: int = 16
    trace_feats: int = 16

    # === 重要な構造パラメータ ===
    # Trace: 無限加算ではなく「指数移動平均」にする
    alpha_trace: float = 0.1   # 0<alpha<=1 : 新しい入力への追従率
    # Self: h_{t+1} = (1-α)h_t + α * target を基本形にする
    alpha_h: float = 0.15
    # Self の飽和スケール（ホメオスタシス）
    h_scale: float = 4.0

    # Affect 計算用（報酬＋ショック）
    w_reward: float = 1.0
    w_shock: float = 0.5

    # Self への寄与（距離・情動・Trace）
    w_h_from_dist: float = 0.8
    w_h_from_affect: float = 0.6
    w_h_from_trace: float = 0.3

    # policy（行動決定）
    policy_k_h: float = 0.7
    policy_k_dist: float = 1.0
    policy_noise_std: float = 0.02

    seed: int = 42


# ===========================================================
# 2. 1D 環境
# ===========================================================

class MiniEnv1D:
    def __init__(self, cfg: SIAConfig1D):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.x = None
        self.goal = cfg.goal_pos
        self.step_count = 0

    def reset(self):
        self.x = 0.1  # 左側から開始
        self.step_count = 0
        return {"x": self.x, "goal": self.goal}

    def step(self, action: float):
        self.x += action
        self.x = max(self.cfg.x_min, min(self.cfg.x_max, self.x))
        self.step_count += 1

        dist = abs(self.goal - self.x)
        # 目標に近いほど 0 に近づく（遠いほど負）
        reward = -dist

        done = False
        if self.step_count >= self.cfg.max_steps:
            done = True
        if dist < 0.02:
            done = True

        obs = {"x": self.x, "goal": self.goal}
        info = {"dist": dist}
        return obs, reward, done, info


# ===========================================================
# 3. SIA Core: Shock / Affect / Trace(rank-3) / Self / Policy
# ===========================================================

class SIAAgent1D(nn.Module):
    def __init__(self, cfg: SIAConfig1D):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # Self（自己状態）：スカラー
        self.h = torch.tensor(0.0, dtype=torch.float32)

        # Rank-3 Trace tensor: (C, D, F)
        self.trace = torch.zeros(
            cfg.trace_channels, cfg.trace_depth, cfg.trace_feats, dtype=torch.float32
        )

        # 各チャネルの「型」（正規化済み）
        self.register_buffer(
            "basis_shock",
            F.normalize(torch.randn(cfg.trace_depth, cfg.trace_feats), dim=-1),
        )
        self.register_buffer(
            "basis_affect",
            F.normalize(torch.randn(cfg.trace_depth, cfg.trace_feats), dim=-1),
        )
        self.register_buffer(
            "basis_self",
            F.normalize(torch.randn(cfg.trace_depth, cfg.trace_feats), dim=-1),
        )

        # 勾配・ショック計算用の履歴
        self.last_obs = None
        self.last_x = None
        self.last_h = None

    def reset_state(self):
        self.h = torch.tensor(0.0, dtype=torch.float32)
        self.trace.zero_()
        self.last_obs = None
        self.last_x = None
        self.last_h = None

    # ---------- Shock ----------
    def compute_shock(self, obs_x: float):
        if self.last_obs is None:
            shock = 0.0
        else:
            shock = float(abs(obs_x - self.last_obs))
        self.last_obs = float(obs_x)
        return shock

    # ---------- Affect ----------
    def compute_affect(self, reward: float, shock: float):
        # reward: 負（遠い）〜0（近い）
        val = self.cfg.w_reward * reward + self.cfg.w_shock * shock
        # [-1, 1] に正規化
        return float(math.tanh(val))

    # ---------- Trace update（EMA 型＝発散しない） ----------
    def update_trace(self, shock: float, affect: float, h_value: float):
        alpha = self.cfg.alpha_trace
        # 古い痕跡を (1-α) 倍：指数移動平均
        self.trace *= (1.0 - alpha)

        # 新規入力を [-1,1] に潰してから basis に乗せる
        s_shock = math.tanh(shock)
        s_affect = affect
        s_self = math.tanh(0.5 * h_value)

        self.trace[0] += alpha * s_shock * self.basis_shock
        self.trace[1] += alpha * s_affect * self.basis_affect
        self.trace[2] += alpha * s_self * self.basis_self

    # ---------- Self update（固定点ダイナミクス） ----------
    def update_self(self, dist_signed: float, affect: float):
        # dist_signed = goal - x （右にゴールがあると正）
        # Self は「右に行きたい/左に行きたい」の傾きとして解釈
        drive_dist = self.cfg.w_h_from_dist * dist_signed
        drive_aff = self.cfg.w_h_from_affect * affect

        # Trace（affect channel）の平均を [-1,1] にクリップ
        trace_aff = float(self.trace[1].mean().item())
        trace_aff = max(-1.0, min(1.0, trace_aff))
        drive_trace = self.cfg.w_h_from_trace * trace_aff

        # Self が向かう「目標値」
        target = drive_dist + drive_aff + drive_trace

        # h_{t+1} = (1-α) h_t + α * target
        alpha = self.cfg.alpha_h
        h_new = (1.0 - alpha) * float(self.h.item()) + alpha * target

        # ホメオスタシス（|h| を h_scale 付近に抑える）
        h_scaled = h_new / self.cfg.h_scale
        h_bounded = self.cfg.h_scale * math.tanh(h_scaled)
        self.h = torch.tensor(h_bounded, dtype=torch.float32)

    # ---------- Policy（行動） ----------
    def policy(self, dist_signed: float):
        # h は「方向の好み」、dist は「物理的な誤差」
        h_val = float(self.h.item())
        base = self.cfg.policy_k_h * h_val + self.cfg.policy_k_dist * dist_signed
        action = math.tanh(base)
        noise = random.gauss(0.0, self.cfg.policy_noise_std)
        return float(action + noise)

    # ---------- Self-field 勾配 ----------
    def compute_gradient(self, x: float):
        if self.last_x is None or self.last_h is None:
            g = 0.0
        else:
            dx = float(x - self.last_x)
            if abs(dx) < 1e-5:
                g = 0.0
            else:
                dh = float(self.h.item() - self.last_h)
                g = dh / dx
        self.last_x = float(x)
        self.last_h = float(self.h.item())
        return g

    # ---------- 1 ステップ ----------
    def step(self, obs_x: float, goal_x: float, reward_prev: float):
        dist_signed = float(goal_x - obs_x)

        # 1) shock
        shock = self.compute_shock(obs_x)

        # 2) affect
        affect = self.compute_affect(reward_prev, shock)

        # 3) trace 更新（EMA）
        self.update_trace(shock, affect, float(self.h.item()))

        # 4) self 更新（固定点へ収束）
        self.update_self(dist_signed, affect)

        # 5) 行動
        action = self.policy(dist_signed)

        # 6) Self-field 勾配
        grad = self.compute_gradient(obs_x)

        # 診断用
        trace_norm_total = float(self.trace.norm().item())
        trace_norm_ch = [
            float(self.trace[c].norm().item())
            for c in range(self.cfg.trace_channels)
        ]

        h_val = float(self.h.item())
        return {
            "action": action,
            "shock": shock,
            "affect": affect,
            "dist_signed": dist_signed,
            "h": h_val,
            "self_grad": grad,
            "trace_norm_total": trace_norm_total,
            "trace_norm_ch": trace_norm_ch,
        }


# ===========================================================
# 4. シミュレーション & 可視化
# ===========================================================

def run_episode(cfg: SIAConfig1D):
    env = MiniEnv1D(cfg)
    agent = SIAAgent1D(cfg)
    agent.reset_state()

    obs = env.reset()
    reward_prev = 0.0

    ts = []
    xs = []
    goals = []
    rewards = []
    actions = []
    h_vals = []
    grads = []
    shocks = []
    affects = []
    trace_norms = []
    trace_norms_ch0 = []
    trace_norms_ch1 = []
    trace_norms_ch2 = []

    for t in range(cfg.max_steps):
        x = float(obs["x"])
        goal = float(obs["goal"])

        out = agent.step(x, goal, reward_prev)
        action = out["action"]

        obs, reward, done, info = env.step(action)

        ts.append(t)
        xs.append(x)
        goals.append(goal)
        rewards.append(reward)
        actions.append(action)
        h_vals.append(out["h"])
        grads.append(out["self_grad"])
        shocks.append(out["shock"])
        affects.append(out["affect"])
        trace_norms.append(out["trace_norm_total"])
        trace_norms_ch0.append(out["trace_norm_ch"][0])
        trace_norms_ch1.append(out["trace_norm_ch"][1])
        trace_norms_ch2.append(out["trace_norm_ch"][2])

        reward_prev = reward

        if done:
            break

    return {
        "t": ts,
        "x": xs,
        "goal": goals,
        "reward": rewards,
        "action": actions,
        "h": h_vals,
        "grad": grads,
        "shock": shocks,
        "affect": affects,
        "trace_norm": trace_norms,
        "trace_norm_ch0": trace_norms_ch0,
        "trace_norm_ch1": trace_norms_ch1,
        "trace_norm_ch2": trace_norms_ch2,
    }


def plot_results(log):
    t = log["t"]
    x = log["x"]
    goal = log["goal"]
    reward = log["reward"]
    action = log["action"]
    h = log["h"]
    grad = log["grad"]
    shock = log["shock"]
    affect = log["affect"]
    trace_norm = log["trace_norm"]
    trace_norm_ch0 = log["trace_norm_ch0"]
    trace_norm_ch1 = log["trace_norm_ch1"]
    trace_norm_ch2 = log["trace_norm_ch2"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 1) 位置とゴール
    ax = axes[0]
    ax.plot(t, x, label="x (position)")
    ax.plot(t, goal, "--", label="goal")
    ax.set_title("Position vs Goal")
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.legend()

    # 2) Self と 勾配
    ax = axes[1]
    ax.plot(t, h, label="h (Self)")
    ax.set_ylabel("h")
    ax2 = ax.twinx()
    ax2.plot(t, grad, "r--", label="∂h/∂x")
    ax.set_title("Self h and Self-field gradient")
    ax.set_xlabel("time")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # 3) Reward と Action
    ax = axes[2]
    ax.plot(t, reward, label="reward")
    ax2 = ax.twinx()
    ax2.plot(t, action, "g--", label="action")
    ax.set_title("Reward and Action")
    ax.set_xlabel("time")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # 4) Shock と Affect
    ax = axes[3]
    ax.plot(t, shock, label="shock")
    ax2 = ax.twinx()
    ax2.plot(t, affect, "m--", label="affect")
    ax.set_title("Shock and Affect")
    ax.set_xlabel("time")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # 5) Trace ノルム
    ax = axes[4]
    ax.plot(t, trace_norm, label="trace total")
    ax.plot(t, trace_norm_ch0, label="ch0 (shock)")
    ax.plot(t, trace_norm_ch1, label="ch1 (affect)")
    ax.plot(t, trace_norm_ch2, label="ch2 (self)")
    ax.set_title("Trace Norm Dynamics")
    ax.set_xlabel("time")
    ax.set_ylabel("‖Trace‖")
    ax.legend()

    # 6) 位相図: |h| vs |affect|
    ax = axes[5]
    h_abs = [abs(v) for v in h]
    affect_abs = [abs(v) for v in affect]
    ax.plot(h_abs, affect_abs, "-o", markersize=2)
    ax.set_title("Phase: |h| vs |affect|")
    ax.set_xlabel("|h|")
    ax.set_ylabel("|affect|")

    # 7) 位相図: |h| vs ‖Trace‖
    ax = axes[6]
    ax.plot(h_abs, trace_norm, "-o", markersize=2)
    ax.set_title("Phase: |h| vs ‖Trace‖")
    ax.set_xlabel("|h|")
    ax.set_ylabel("‖Trace‖")

    # 8) Self-field gradient vs position
    ax = axes[7]
    sc = ax.scatter(x, grad, c=t, cmap="viridis")
    ax.set_title("Self-field gradient over position")
    ax.set_xlabel("x (position)")
    ax.set_ylabel("∂h/∂x")
    fig = ax.get_figure()
    fig.colorbar(sc, ax=ax, label="time")

    plt.tight_layout()
    plt.show()


def main():
    cfg = SIAConfig1D()
    log = run_episode(cfg)
    print("Final x, h:", log["x"][-1], log["h"][-1])
    plot_results(log)


if __name__ == "__main__":
    main()
