import math
import statistics
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Config
# ============================================================

@dataclass
class MiniSIAConfig:
    obs_dim: int = 2
    h_dim: int = 8
    affect_dim: int = 4

    # Trace parameters
    trace_decay: float = 0.95
    trace_lr: float = 0.1

    # Self parameters
    h_lr: float = 0.05
    max_h_norm: float = 5.0
    max_action: float = 0.2

    # gains (to be overwritten inside scan)
    alpha_past: float = 0.4
    beta_present: float = 0.8
    gamma_future: float = 0.6

    # simulation
    steps_per_episode: int = 80
    n_episodes: int = 1


# ============================================================
# 2. Environment
# ============================================================

class MiniLineEnv:
    def __init__(self, max_steps: int = 80):
        self.max_steps = max_steps
        self.x = 0.0
        self.goal = 1.0
        self.step_count = 0

    def reset(self):
        self.x = float(torch.rand(1).item() * 0.5)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return torch.tensor([self.x, self.goal], dtype=torch.float32)

    def step(self, action: float):
        self.step_count += 1
        self.x += action
        self.x = max(0.0, min(1.0, self.x))

        obs = self._get_obs()
        dist = abs(self.goal - self.x)
        reward = -dist

        done = (dist < 0.01) or (self.step_count >= self.max_steps)
        return obs, reward, done, {"dist_to_goal": dist}


# ============================================================
# 3. Modules
# ============================================================

class ShockModule(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, obs, prev_obs):
        if prev_obs is None:
            return torch.zeros_like(obs)
        return obs - prev_obs


class AffectModule(nn.Module):
    def __init__(self, obs_dim: int, affect_dim: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, affect_dim)

    def forward(self, shock):
        return torch.tanh(self.fc(shock))


class TraceModule(nn.Module):
    def __init__(self, affect_dim, h_dim, cfg: MiniSIAConfig):
        super().__init__()
        self.decay = cfg.trace_decay
        self.lr = cfg.trace_lr

    def forward(self, affect, h, trace_prev):
        imprint = torch.ger(affect, h)
        return self.decay * trace_prev + self.lr * imprint


class SelfModule(nn.Module):
    """
    h(t+1) = h(t) + h_lr * tanh( α * Past + β * Present + γ * Future )
    """
    def __init__(self, affect_dim, h_dim, cfg: MiniSIAConfig):
        super().__init__()
        trace_flat = affect_dim * h_dim
        self.cfg = cfg

        self.fc_past = nn.Linear(trace_flat, h_dim)
        self.fc_present = nn.Linear(affect_dim, h_dim)
        self.fc_future = nn.Linear(1, h_dim)
        self.fc_action = nn.Linear(h_dim, 1)

    def forward(self, affect, trace, future_signal, h_prev):
        trace_flat = trace.view(-1)
        past = self.fc_past(trace_flat)
        present = self.fc_present(affect)
        future = self.fc_future(future_signal)

        u = (
            self.cfg.alpha_past * past +
            self.cfg.beta_present * present +
            self.cfg.gamma_future * future
        )

        delta_h = torch.tanh(u)
        h = h_prev + self.cfg.h_lr * delta_h

        # norm constraint
        norm_h = torch.norm(h)
        if norm_h > self.cfg.max_h_norm:
            h = h * (self.cfg.max_h_norm / (norm_h + 1e-8))

        action = torch.tanh(self.fc_action(h))[0] * self.cfg.max_action
        return h, action


# ============================================================
# 4. Agent
# ============================================================

class MiniSIAAgent:
    def __init__(self, cfg: MiniSIAConfig):
        self.cfg = cfg
        self.shock_module = ShockModule(cfg.obs_dim)
        self.affect_module = AffectModule(cfg.obs_dim, cfg.affect_dim)
        self.trace_module = TraceModule(cfg.affect_dim, cfg.h_dim, cfg)
        self.self_module = SelfModule(cfg.affect_dim, cfg.h_dim, cfg)

        self.reset()

    def reset(self):
        self.h = torch.zeros(self.cfg.h_dim)
        self.trace = torch.zeros(self.cfg.affect_dim, self.cfg.h_dim)
        self.prev_obs = None

    def step(self, obs):
        # shock
        shock = self.shock_module(obs, self.prev_obs)

        # affect
        affect = self.affect_module(shock)

        # trace update
        self.trace = self.trace_module(affect, self.h, self.trace)

        # future signal
        position, goal = obs[0], obs[1]
        future_signal = torch.tensor([goal - position], dtype=torch.float32)

        # self update
        self.h, action = self.self_module(affect, self.trace, future_signal, self.h)
        self.prev_obs = obs.clone()

        metrics = {
            "h_norm": torch.norm(self.h).item(),
            "trace_norm": torch.norm(self.trace).item(),
            "affect_norm": torch.norm(affect).item(),
            "action": float(action.item()),
        }
        return action.item(), metrics


# ============================================================
# 5. Run single simulation
# ============================================================

def run_single_sim(cfg: MiniSIAConfig):
    env = MiniLineEnv(cfg.steps_per_episode)
    agent = MiniSIAAgent(cfg)

    obs = env.reset()
    agent.reset()

    h_vals = []
    trace_vals = []
    affect_vals = []
    actions = []

    for step in range(cfg.steps_per_episode):
        with torch.no_grad():
            action, metrics = agent.step(obs)

        obs, reward, done, info = env.step(action)

        h_vals.append(metrics["h_norm"])
        trace_vals.append(metrics["trace_norm"])
        affect_vals.append(metrics["affect_norm"])
        actions.append(metrics["action"])

        if done:
            break

    max_h = max(h_vals)
    max_trace = max(trace_vals)
    var_action = statistics.pvariance(actions) if len(actions) > 1 else 0.0
    final_dist = info["dist_to_goal"]
    steps = step + 1

    return {
        "max_h": max_h,
        "max_trace": max_trace,
        "var_action": var_action,
        "steps": steps,
        "final_dist": final_dist,
    }


# ============================================================
# 6. Scan parameter space
# ============================================================

def scan_sia_parameter_space():
    cfg = MiniSIAConfig()

    alphas = [0.0, 0.3, 0.6, 1.0, 1.5]
    betas  = [0.0, 0.3, 0.6, 1.0, 1.5]
    gammas = [0.0, 0.3, 0.6, 1.0, 1.5]

    results = {}

    for a in alphas:
        for b in betas:
            for g in gammas:
                cfg.alpha_past = a
                cfg.beta_present = b
                cfg.gamma_future = g

                r = run_single_sim(cfg)
                results[(a, b, g)] = r
                print(f"[α={a}, β={b}, γ={g}]  ->  {r}")

    os.makedirs("figs", exist_ok=True)

    # create heatmaps for each α
    for i, a in enumerate(alphas):
        heat = np.zeros((len(betas), len(gammas)))

        for ib, b in enumerate(betas):
            for ig, g in enumerate(gammas):
                r = results[(a, b, g)]
                instability = (
                    r["max_h"] +
                    r["var_action"] * 10.0 +
                    r["max_trace"] * 0.3
                )
                heat[ib, ig] = instability

        plt.figure(figsize=(6, 5))
        plt.imshow(heat, cmap="hot", origin="lower", extent=[0, 1, 0, 1])
        plt.colorbar(label="Instability")
        plt.title(f"Instability Map (α={a})")
        plt.xlabel("γ (future)")
        plt.ylabel("β (present)")
        plt.tight_layout()
        plt.savefig(f"figs/instability_alpha_{a}.png")
        plt.close()

    print("\n=== Saved heatmaps to ./figs/ ===")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    scan_sia_parameter_space()
