# src/identity/visualization/visualize_self_evolution.py
# -*- coding: utf-8 -*-

"""
Visualization: Evolution of Self-Space Geometry over Time

目的：
    Imprint によって、Self-space が
        - どのように変形し、
        - Identityの重心（Trace / Self-center）がどう移動し、
        - Self-axes がどう形成・強化され、
        - 心の井戸 (Attractor basin) がどう成長するか
    を、「時間発展」として可視化する。

理論上重要な観点：
    ✔ Identityは点ではなく、進化する幾何構造である
    ✔ Potential / Curvature の変化こそが「経験が自己を変形させた証拠」
    ✔ Traceは「記憶」ではなく、「持続する自己変形の核」
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation


def project_to_2d(vectors: torch.Tensor):
    """(N,d) → (N,2) にPCA射影"""
    return PCA(n_components=2).fit_transform(vectors.detach().cpu().numpy())


def animate_self_evolution(engine, traces, interval=800, save_path=None):
    """
    時系列のTraceを順にengineへ適用し、
    Self-spaceのPotential変形 & Trace軌道 & Self-center進化を可視化する。

    Args:
        engine: ImprintGeometryEngine
        traces: List[torch.Tensor]  Trace (Identity_core) at each time step
        interval: frame interval (ms)
        save_path: save as .mp4 or .gif
    """

    coords = engine.self_space.coords
    coords_2d = project_to_2d(coords)
    fig, ax = plt.subplots(figsize=(7,6))

    trace_history = []  # to log trajectory

    def update(frame):
        ax.clear()

        # === 1) Apply new trace (Identity update) ===
        current_trace = traces[frame]
        engine.trace.step(current_trace)
        engine.self_space.update_potential(engine.trace.trace)
        engine.self_space.update_metric_from_potential()
        engine.self_space.update_curvature_from_potential_laplacian()

        # === 2) Recompute parameters ===
        potential = engine.self_space.potential.detach().cpu().numpy()
        curvature = engine.self_space.curvature.detach().cpu().numpy()
        self_center = engine.trace.trace.detach().cpu().numpy()

        # PCA projection for Self-center
        self_center_2d = PCA(n_components=2).fit_transform(
            np.vstack([coords.detach().cpu().numpy(), self_center])
        )[-1]

        trace_history.append(self_center_2d)

        # === 3) Plot Potential landscape ===
        sc = ax.scatter(
            coords_2d[:,0], coords_2d[:,1],
            c=potential,
            cmap="viridis",
            s=50,
            alpha=0.8
        )

        # === 4) Plot Trace / Identity core (Self-center) ===
        ax.scatter(self_center_2d[0], self_center_2d[1],
                   c="red", marker="*", s=300, label="Self-center")

        # === 5) Plot trajectory ===
        if len(trace_history) > 1:
            path = np.array(trace_history)
            ax.plot(path[:,0], path[:,1], 'r--', linewidth=2, alpha=0.7)

        ax.set_title(f"Self-Space Evolution — Step {frame+1}/{len(traces)}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.legend()
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(traces), interval=interval, repeat=False)

    if save_path:
        anim.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'imagemagick')

    plt.show()
