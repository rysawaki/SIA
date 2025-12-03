import math
import statistics
from dataclasses import dataclass

import torch
import torch.nn as nn


# ============================================================
# 1. Config
# ============================================================

@dataclass
class MiniSIAConfig:
    # 次元
    obs_dim: int = 2        # 1D環境: [position, goal]
    h_dim: int = 8          # Self の次元
    affect_dim: int = 4     # Affect の次元

    # Trace の更新パラメータ
    trace_decay: float = 0.95
    trace_lr: float = 0.1

    # Self 更新・制約
    h_lr: float = 0.05
    max_h_norm: float = 5.0
    max_trace_norm: float = 10.0
    max_affect_norm: float = 3.0

    # 行動
    max_action: float = 0.2

    # シミュレーション
    steps_per_episode: int = 200
    n_episodes: int = 3

    # 安定判定用の「許容閾値」（少し余裕を持たせる）
    stable_h_norm_threshold: float = 6.0
    stable_trace_norm_threshold: float = 12.0
    stable_affect_norm_threshold: float = 4.0
    stable_action_var_threshold: float = 0.05  # 行動の分散


# ============================================================
# 2. 1D Mini 環境
# ============================================================

class MiniLineEnv:
    """
    0〜1 の線分上を動くエージェント。
    goal = 1.0 に近づくと報酬が高くなる。
    """
    def __init__(self, max_steps: int = 200):
        self.x = 0.0
        self.goal = 1.0
        self.step_count = 0
        self.max_steps = max_steps

    def reset(self):
        self.x = torch.rand(1).item() * 0.5  # 0〜0.5 のどこかからスタート
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # obs = [現在位置, ゴール位置]
        return torch.tensor([self.x, self.goal], dtype=torch.float32)

    def step(self, action: float):
        self.step_count += 1
        # 1D 上で移動
        self.x += float(action)
        self.x = max(0.0, min(1.0, self.x))

        obs = self._get_obs()
        dist = abs(self.goal - self.x)
        reward = -dist

        done = False
        if dist < 0.01:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {"dist_to_goal": dist}
        return obs, reward, done, info


# ============================================================
# 3. SIA Mini Modules
# ============================================================

class ShockModule(nn.Module):
    """
    shock = obs - prev_obs
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, prev_obs: torch.Tensor | None):
        if prev_obs is None:
            return torch.zeros_like(obs)
        return obs - prev_obs


class AffectModule(nn.Module):
    """
    Affect = tanh(W_s * shock + b)
    """
    def __init__(self, obs_dim: int, affect_dim: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, affect_dim)

    def forward(self, shock: torch.Tensor):
        return torch.tanh(self.fc(shock))


class TraceModule(nn.Module):
    """
    Trace: rank-2 (affect_dim × h_dim) の行列として実装。
    T(t+1) = decay * T(t) + lr * (affect ⊗ h)
    """
    def __init__(self, affect_dim: int, h_dim: int, cfg: MiniSIAConfig):
        super().__init__()
        self.affect_dim = affect_dim
        self.h_dim = h_dim
        self.decay = cfg.trace_decay
        self.lr = cfg.trace_lr

    def forward(self, affect: torch.Tensor, h: torch.Tensor, trace_prev: torch.Tensor):
        # outer product: (affect_dim, h_dim)
        imprint = torch.ger(affect, h)  # affect ⊗ h
        trace = self.decay * trace_prev + self.lr * imprint
        return trace


class SelfModule(nn.Module):
    """
    Self h を更新し、行動を生成する。
    h(t+1) = h(t) + h_lr * tanh( W_a * affect + W_t * vec(trace) )
    action = tanh(w_act * h) * max_action
    """
    def __init__(self, affect_dim: int, h_dim: int, cfg: MiniSIAConfig):
        super().__init__()
        self.h_dim = h_dim
        self.cfg = cfg
        trace_flat_dim = affect_dim * h_dim

        self.fc_affect = nn.Linear(affect_dim, h_dim)
        self.fc_trace = nn.Linear(trace_flat_dim, h_dim)
        self.fc_action = nn.Linear(h_dim, 1)

    def forward(self, affect: torch.Tensor, trace: torch.Tensor, h_prev: torch.Tensor):
        trace_flat = trace.view(-1)
        # 入力総和
        u = self.fc_affect(affect) + self.fc_trace(trace_flat)
        delta_h = torch.tanh(u)
        h = h_prev + self.cfg.h_lr * delta_h

        # ノルム制御
        norm_h = torch.norm(h)
        if norm_h > self.cfg.max_h_norm:
            h = h * (self.cfg.max_h_norm / (norm_h + 1e-8))

        # 行動
        action = torch.tanh(self.fc_action(h))[0] * self.cfg.max_action
        return h, action


# ============================================================
# 4. SIA Agent (Shock → Affect → Trace → Self)
# ============================================================

class MiniSIAAgent:
    def __init__(self, cfg: MiniSIAConfig):
        self.cfg = cfg

        self.shock_module = ShockModule(cfg.obs_dim)
        self.affect_module = AffectModule(cfg.obs_dim, cfg.affect_dim)
        self.trace_module = TraceModule(cfg.affect_dim, cfg.h_dim, cfg)
        self.self_module = SelfModule(cfg.affect_dim, cfg.h_dim, cfg)

        # 内部状態
        self.h = torch.zeros(cfg.h_dim)
        self.trace = torch.zeros(cfg.affect_dim, cfg.h_dim)
        self.prev_obs: torch.Tensor | None = None

    def reset(self):
        self.h = torch.zeros(self.cfg.h_dim)
        self.trace = torch.zeros(self.cfg.affect_dim, self.cfg.h_dim)
        self.prev_obs = None

    def step(self, obs: torch.Tensor):
        # 1. Shock
        shock = self.shock_module(obs, self.prev_obs)

        # 2. Affect
        affect = self.affect_module(shock)

        # 3. Trace
        self.trace = self.trace_module(affect, self.h, self.trace)

        # 4. Self + Action
        self.h, action = self.self_module(affect, self.trace, self.h)

        # 次のステップのために保存
        self.prev_obs = obs.clone()

        # 安定判定用のログ
        metrics = {
            "h_norm": torch.norm(self.h).item(),
            "trace_norm": torch.norm(self.trace).item(),
            "affect_norm": torch.norm(affect).item(),
            "action": float(action.item()),
        }
        return action.item(), metrics


# ============================================================
# 5. Stability Monitor
# ============================================================

class StabilityMonitor:
    """
    各ステップの内部ノルムと行動を記録し、
    シミュレーション後に「安定かどうか」を判定する。
    """

    def __init__(self, cfg: MiniSIAConfig):
        self.cfg = cfg
        self.h_norms: list[float] = []
        self.trace_norms: list[float] = []
        self.affect_norms: list[float] = []
        self.actions: list[float] = []

    def record(self, metrics: dict):
        self.h_norms.append(metrics["h_norm"])
        self.trace_norms.append(metrics["trace_norm"])
        self.affect_norms.append(metrics["affect_norm"])
        self.actions.append(metrics["action"])

    def report(self):
        if not self.h_norms:
            print("No data recorded.")
            return

        max_h = max(self.h_norms)
        max_trace = max(self.trace_norms)
        max_affect = max(self.affect_norms)
        action_var = statistics.pvariance(self.actions) if len(self.actions) > 1 else 0.0

        # 閾値判定
        stable_h = max_h <= self.cfg.stable_h_norm_threshold
        stable_trace = max_trace <= self.cfg.stable_trace_norm_threshold
        stable_affect = max_affect <= self.cfg.stable_affect_norm_threshold
        stable_action = action_var <= self.cfg.stable_action_var_threshold

        is_stable = stable_h and stable_trace and stable_affect and stable_action

        print("\n================ Stability Report ================")
        print(f"Steps total           : {len(self.h_norms)}")
        print(f"max ||h||             : {max_h:.4f}  (threshold {self.cfg.stable_h_norm_threshold})")
        print(f"max ||Trace||         : {max_trace:.4f}  (threshold {self.cfg.stable_trace_norm_threshold})")
        print(f"max ||Affect||        : {max_affect:.4f}  (threshold {self.cfg.stable_affect_norm_threshold})")
        print(f"Var(action)           : {action_var:.6f} (threshold {self.cfg.stable_action_var_threshold})")
        print("--------------------------------------------------")
        print(f"Stable h?             : {stable_h}")
        print(f"Stable trace?         : {stable_trace}")
        print(f"Stable affect?        : {stable_affect}")
        print(f"Stable action?        : {stable_action}")
        print("--------------------------------------------------")
        print(f"Overall SIA stability : {is_stable}")
        print("==================================================\n")


# ============================================================
# 6. Main: Mini-SIA Simulation + Stability Check
# ============================================================

def run_mini_sia_simulation():
    cfg = MiniSIAConfig()

    env = MiniLineEnv(max_steps=cfg.steps_per_episode)
    agent = MiniSIAAgent(cfg)
    monitor = StabilityMonitor(cfg)

    for ep in range(cfg.n_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        ep_reward = 0.0

        for step in range(cfg.steps_per_episode):
            with torch.no_grad():
                action, metrics = agent.step(obs)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            monitor.record(metrics)

            # 必要なら途中でデバッグ出力
            # if step % 50 == 0:
            #     print(f"[Ep {ep} Step {step}] h_norm={metrics['h_norm']:.3f}, "
            #           f"trace_norm={metrics['trace_norm']:.3f}, "
            #           f"affect_norm={metrics['affect_norm']:.3f}, "
            #           f"action={metrics['action']:.3f}, "
            #           f"dist={info['dist_to_goal']:.3f}")

            if done:
                break

        print(f"Episode {ep} finished in {step + 1} steps, total reward = {ep_reward:.3f}")

    # シミュレーション全体の安定性をレポート
    monitor.report()


if __name__ == "__main__":
    run_mini_sia_simulation()
