import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from src_v1.v6 import SIAConfig_v6_1, SIASystem_v6_1


# ============================================================
# Re-import v6.1 Classes (Assuming definition above)
# ============================================================
# (TraceGenome, MetabolicKernel_v6_1, GeometricSelfManifold, SIASystem_v6_1)
# ここでは再定義せず、先ほどの定義済みクラスを使う前提でRunnerを書きます。

# ============================================================
# Experiment Logic
# ============================================================

def run_experiment_v6_metabolism():
    config = SIAConfig_v6_1()
    # Metabolic Rate調整 (速すぎず、遅すぎず)
    config.metabolic_rate = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIASystem_v6_1(config).to(device)

    # Kernelの学習率 (Operatorの進化速度)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # 実験用サブセット
    subset_indices = torch.arange(5000)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(
        f"{'Epoch':<5} | {'Phase':<8} | {'Acc':<6} | {'Loss':<8} | {'|Genome|':<8} | {'|G|':<8} | {'|μ|':<8} | {'ΔSpace':<8}")
    print("-" * 80)

    history = {
        'G_norm': [], 'mu_norm': [], 'genome_norm': [],
        'space_diff': [], 'acc': []
    }

    # 直前の空間パラメータを保持（変化量計算用）
    prev_G = model.self_manifold.current_G.detach().clone()
    prev_mu = model.self_manifold.current_mu.detach().clone()

    for epoch in range(15):
        # Phase Control: Normal -> Trauma (Rotate 90) -> Recovery
        if epoch < 6:
            phase = "Normal"
            angle = 0
        elif epoch < 11:
            phase = "TRAUMA"
            angle = 1
        else:
            phase = "Recov"
            angle = 0

        model.train()
        total = 0
        correct = 0
        epoch_loss = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if angle > 0:
                imgs = torch.rot90(imgs, angle, [2, 3])

            optimizer.zero_grad()

            # Forward (Includes Metabolism)
            logits, loss = model(imgs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        # Epoch Metrics
        acc = 100 * correct / total
        avg_loss = epoch_loss / (i + 1)

        # Geometry Stats
        curr_genome = model.trace.read()
        curr_G = model.self_manifold.current_G
        curr_mu = model.self_manifold.current_mu

        genome_norm = curr_genome.norm().item()
        G_norm = curr_G.norm().item()
        mu_norm = curr_mu.norm().item()

        # 空間の変化量 (Metabolic Flux)
        # ||ΔG|| + ||Δμ||
        delta_G = (curr_G - prev_G).norm().item()
        delta_mu = (curr_mu - prev_mu).norm().item()
        space_diff = delta_G + delta_mu

        # Update prev state
        prev_G = curr_G.detach().clone()
        prev_mu = curr_mu.detach().clone()

        # Log
        history['G_norm'].append(G_norm)
        history['mu_norm'].append(mu_norm)
        history['space_diff'].append(space_diff)
        history['acc'].append(acc)

        print(
            f"{epoch:<5} | {phase:<8} | {acc:<6.2f} | {avg_loss:<8.4f} | {genome_norm:<8.4f} | {G_norm:<8.4f} | {mu_norm:<8.4f} | {space_diff:<8.4f}")

    return history


if __name__ == "__main__":
    hist = run_experiment_v6_metabolism()

    # Plot Geometry Evolution
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Space Norm (|G|, |μ|)', color='tab:blue')
    ax1.plot(hist['G_norm'], label='Metric Norm |G|', color='tab:blue', linestyle='-')
    ax1.plot(hist['mu_norm'], label='Center Norm |μ|', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Space Flux (ΔSpace)', color='tab:red')
    ax2.plot(hist['space_diff'], label='Metabolic Flux', color='tab:red', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Phase Markers
    plt.axvline(x=6, color='k', linestyle=':', label='Trauma Start')
    plt.axvline(x=11, color='k', linestyle=':', label='Recovery Start')

    plt.title("SIA v6.1: Metabolic Evolution of Self-Space")
    fig.tight_layout()
    plt.show()