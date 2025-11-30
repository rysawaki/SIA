# src/identity/visualization/visualize_self_geometry.py
# -*- coding: utf-8 -*-
"""
Visualization: Self-Space Geometry (Potential & Curvature)

目的：
    Imprint によって Self-space がどのように歪み、
    「心の井戸（Attractor）」が形成されるかを可視化する。

可視化対象：
    ✔ Self-space points (coords)
    ✔ Potential V(s)
    ✔ Curvature distortion K(s)
    ✔ Trace / Self-center

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def project_to_2d(vectors: torch.Tensor):
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors.detach().cpu().numpy())


def visualize_self_potential(engine, title="Self-Space Potential Landscape"):
    """
    Self-space の Potential V(s) を PCA投影し、カラー分布で描画する。

    Args:
        engine (ImprintGeometryEngine)
    """
    coords = engine.self_space.coords              # (N,d)
    potential = engine.self_space.potential.cpu()  # (N,)
    self_center = engine.self_center.cpu()

    coords_2d = project_to_2d(coords)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords_2d[:,0], coords_2d[:,1],
                     c=potential,
                     cmap="viridis",
                     s=45, alpha=0.85)

    plt.scatter(self_center[0], self_center[1],
                c="red", marker="*", s=200, label="Self-center")

    plt.colorbar(sc, label="Potential V(s)")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_self_curvature(engine, title="Self-Space Curvature Landscape"):
    """
    Self-space の曲率 K(s) を PCA空間で3D散布図として可視化する。

    Args:
        engine (ImprintGeometryEngine)
    """
    coords = engine.self_space.coords
    curvature = engine.self_space.curvature.cpu()
    coords_2d = project_to_2d(coords)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        coords_2d[:,0],
        coords_2d[:,1],
        curvature,
        c=curvature,
        cmap="plasma",
        s=50,
        alpha=0.85,
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Curvature")
    fig.colorbar(sc, ax=ax, label="Curvature")
    plt.show()


def visualize_self_geometry(engine, title="Self-Space Geometry Overview"):
    """
    Potential と Curvature を両方重ねた「心の井戸」の近似形を描画する。
    """
    coords = engine.self_space.coords
    coords_2d = project_to_2d(coords)
    potential = engine.self_space.potential.cpu()
    curvature = engine.self_space.curvature.cpu()
    self_center = engine.self_center.cpu()

    fig, ax = plt.subplots(figsize=(8,6))

    sc = ax.scatter(
        coords_2d[:,0], coords_2d[:,1],
        c=potential,
        cmap="viridis",
        s=(curvature.abs().numpy() * 80) + 30,
        alpha=0.75
    )

    ax.scatter(self_center[0], self_center[1],
               c="red", marker="*", s=240, label="Self-center")

    plt.colorbar(sc, label="Potential V(s)")
    plt.title(title + "\n(Size reflects curvature magnitude)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
