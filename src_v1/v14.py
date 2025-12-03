import math
import statistics
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. Mini-SIA Config
# ============================================================

@dataclass
class MiniSIAConfig:
    obs_dim: int = 2
    h_dim: int = 8
    affect_dim: int = 4

    trace_decay: float = 0.95
    trace_lr: float = 0.1

    h_lr: float = 0.05
    max_h_norm: float = 5.0
    max_action: float = 0.2

    alpha_past: float = 0.4
    beta_present: float = 0.8
    gamma_future: float = 0.6

    steps_per_episode: int = 80


# ============================================================
# 2. Environment
# ============================================================

class MiniLineEnv:
    def __init__(self, max_steps: int = 80):
        self.max_steps = max_steps

    def reset(self):
        self.x = float(torch.rand(1).item() * 0.5)
        self.goal = 1.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return torch.tensor([self.x, self.goal], dtype=torch.float32)

    def step(self, action: float):
        self.step_count += 1
        self.x += float(action)
        self.x = max(0.0, min(1.0, self.x))

        obs = self._get_obs()
        dist = abs(self.goal - self.x)
        reward = -dist

        done = (dist < 0.01) or (self.step_count >= self.max_steps)
        info = {"dist": dist}
        return obs, reward, done, info


# ============================================================
# 3. SIA Modules
# ============================================================

class Shock(nn.Module):
    def forward(self, obs, prev):
        return torch.zeros_like(obs) if prev is None else obs - prev


class Affect(nn.Module):
    def __init__(self, obs_dim, affect_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, affect_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class TraceM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, aff, h, T):
        return self.cfg.trace_decay * T + self.cfg.trace_lr * torch.ger(aff, h)


class SelfM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc_p = nn.Linear(cfg.affect_dim * cfg.h_dim, cfg.h_dim)
        self.fc_a = nn.Linear(cfg.affect_dim, cfg.h_dim)
        self.fc_f = nn.Linear(1, cfg.h_dim)
        self.fc_act = nn.Linear(cfg.h_dim, 1)

    def forward(self, aff, T, fut, h_prev):
        trace_flat = T.view(-1)
        u = (
            self.cfg.alpha_past * self.fc_p(trace_flat)
            + self.cfg.beta_present * self.fc_a(aff)
            + self.cfg.gamma_future * self.fc_f(fut)
        )
        dh = torch.tanh(u)
        h = h_prev + self.cfg.h_lr * dh
        n = h.norm()
        if n > self.cfg.max_h_norm:
            h = h * self.cfg.max_h_norm / (n + 1e-8)

        act = torch.tanh(self.fc_act(h))[0] * self.cfg.max_action
        return h, act


# ============================================================
# 4. Agent + Single Episode
# ============================================================

class MiniSIAAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sh = Shock()
        self.af = Affect(cfg.obs_dim, cfg.affect_dim)
        self.tr = TraceM(cfg)
        self.se = SelfM(cfg)
        self.reset()

    def reset(self):
        self.h = torch.zeros(self.cfg.h_dim)
        self.T = torch.zeros(self.cfg.affect_dim, self.cfg.h_dim)
        self.prev = None

    def step(self, obs):
        shock = self.sh(obs, self.prev)
        aff = self.af(shock)
        self.T = self.tr(aff, self.h, self.T)

        fut = torch.tensor([obs[1] - obs[0]])
        self.h, act = self.se(aff, self.T, fut, self.h)

        self.prev = obs.clone()

        return float(act), self.h.norm().item(), self.T.norm().item()


def run_episode(cfg: MiniSIAConfig):
    env = MiniLineEnv(cfg.steps_per_episode)
    agent = MiniSIAAgent(cfg)

    obs = env.reset()
    agent.reset()

    h_vals = []
    T_vals = []
    acts = []

    for step in range(cfg.steps_per_episode):
        with torch.no_grad():
            act, h_norm, T_norm = agent.step(obs)

        obs, _, done, info = env.step(act)

        h_vals.append(h_norm)
        T_vals.append(T_norm)
        acts.append(act)

        if done:
            break

    max_h = max(h_vals)
    max_T = max(T_vals)
    var_a = statistics.pvariance(acts) if len(acts) > 1 else 0.0

    I = max_h + 0.3 * max_T + 10 * var_a
    return I


# ============================================================
# 5. 10 エピソード平均版 I
# ============================================================

def run_10_episode_avg_I(alpha, beta, gamma, cfg_base: MiniSIAConfig):
    cfg = MiniSIAConfig(**vars(cfg_base))
    cfg.alpha_past = alpha
    cfg.beta_present = beta
    cfg.gamma_future = gamma

    Is = []
    for _ in range(10):
        I = run_episode(cfg)
        Is.append(I)

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "I_mean": np.mean(Is),
        "I_std": np.std(Is),
    }


# ============================================================
# 6. Scan 全点 (α,β,γ)
# ============================================================

def scan_all_points():
    cfg_base = MiniSIAConfig()
    vals = [0.0, 0.5, 1.0, 1.5]

    results = []

    for a in vals:
        for b in vals:
            for g in vals:
                r = run_10_episode_avg_I(a, b, g, cfg_base)
                results.append(r)
                print(f"(α={a}, β={b}, γ={g})  →  I_mean={r['I_mean']:.3f}  (std={r['I_std']:.3f})")

    return results


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    results = scan_all_points()

    # 結果をテーブル形式で確認したい場合
    print("\n=== Final Table (10-episode averaged I) ===")
    for r in results:
        print(r)
