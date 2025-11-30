# ============================
# file: visualize_selfspace.py
# Visualization: Query Deformation by Self-conditioning and Structure of Self-Axes
# ============================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def project_to_2d(vectors: torch.Tensor):
    """
    Projects a set of vectors (N, d) into 2D space using PCA.

    Args:
        vectors (torch.Tensor): Input vectors with shape (N, d).

    Returns:
        numpy.ndarray: Projected vectors with shape (N, 2).
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors.detach().cpu().numpy())


def visualize_query_deformation(Q, Q_cond, axes=None, save_path=None):
    """
    Visualizes the geometric deformation of the Query caused by Self-conditioning.
    This plot helps analyze how the 'Self' biases or distorts the input Query.

    Parameters:
        Q (torch.Tensor):      (d,) or (N, d)
            The original input query (Raw perception/Objective data).
        Q_cond (torch.Tensor): (d,) or (N, d)
            The query after passing through the Self-module (Subjective/Biased perception).
        axes (torch.Tensor):   (k, d) or None
            The internal axes of the Self (e.g., Principal components of the Self-matrix).
            Optional.
        save_path (str):       Path to save the figure. If None, it just shows the plot.
    """
    # Ensure batch dimension exists: (d,) -> (1, d)
    if Q.dim() == 1:
        Q = Q.unsqueeze(0)
        Q_cond = Q_cond.unsqueeze(0)

    data = [Q, Q_cond]
    labels = ["Original Q", "Self-conditioned Q"]
    colors = ["blue", "red"]

    if axes is not None:
        data.append(axes)
        labels.append("Self axes")
        colors.append("green")

    # Concatenate all data for unified PCA projection
    data_cat = torch.cat(data, dim=0)  # Shape: (N_total, d)
    proj_2d = project_to_2d(data_cat)  # Shape: (N_total, 2)

    # Define indices for slicing the projected data
    idx_Q = range(len(Q))
    idx_Qc = range(len(Q), len(Q) + len(Q_cond))
    idx_axes = range(len(Q) + len(Q_cond), proj_2d.shape[0])

    plt.figure(figsize=(7, 6))

    # Plot Original and Conditioned Queries
    plt.scatter(proj_2d[idx_Q, 0], proj_2d[idx_Q, 1], label="Original Q", s=60, marker='o', c='blue', alpha=0.7)
    plt.scatter(proj_2d[idx_Qc, 0], proj_2d[idx_Qc, 1], label="Self-conditioned Q", s=60, marker='x', c='red')

    # Visualize the deformation vector (shift) using arrows
    # This represents the "force" or "bias" applied by the Self.
    for i in range(len(Q)):
        plt.arrow(
            proj_2d[idx_Q][i, 0], proj_2d[idx_Q][i, 1],
            proj_2d[idx_Qc][i, 0] - proj_2d[idx_Q][i, 0],
            proj_2d[idx_Qc][i, 1] - proj_2d[idx_Q][i, 1],
            color="gray", alpha=0.6, width=0.003
        )

    # Plot Self axes if provided (shows the structural basis of the Self)
    if axes is not None:
        plt.scatter(proj_2d[idx_axes, 0], proj_2d[idx_axes, 1], label="Self axes", s=80, marker='^', c='green')

    plt.title("Geometric Deformation of Query by Self")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_self_axes(axes, save_path=None):
    """
    Plots the axes within SelfSpace to visualize Diversity vs. Directionality.

    This visualization helps determine if the Self is:
    1. Converging/Fixating (Axes clustered together)
    2. Expanding/Generalizing (Axes spread out)

    Args:
        axes (torch.Tensor): The basis vectors representing the Self state.
        save_path (str): Output path.
    """
    if axes is None or len(axes) == 0:
        print("No Self axes to visualize.")
        return

    proj_2d = project_to_2d(axes)  # (k, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=80, c='green', marker='^')

    for i, (x, y) in enumerate(proj_2d):
        plt.text(x, y, f"Axis {i}", fontsize=9, ha='right')

    plt.title("Distribution of Self Axes (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def compare_before_after(Q_before, Q_after, title="Before vs After Self Update"):
    """
    Visualizes how the Query processing changes before and after a Self Update.
    This is used to verify the 'learning effect' or 'plasticity' of the Self.

    Args:
        Q_before (torch.Tensor): Query conditioned by the Self *before* update.
        Q_after (torch.Tensor):  Query conditioned by the Self *after* update.
        title (str): Plot title.
    """
    data = torch.cat([Q_before, Q_after], dim=0)
    proj_2d = project_to_2d(data)

    N = len(Q_before)
    plt.figure(figsize=(7, 6))

    plt.scatter(proj_2d[:N, 0], proj_2d[:N, 1], label="Before Update", s=60, marker='o', alpha=0.6)
    plt.scatter(proj_2d[N:, 0], proj_2d[N:, 1], label="After Update", s=60, marker='x', c='red')

    # Draw connection lines to show the shift trajectory
    for i in range(N):
        plt.plot(
            [proj_2d[i, 0], proj_2d[N + i, 0]],
            [proj_2d[i, 1], proj_2d[N + i, 1]],
            'k--', alpha=0.3
        )

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()