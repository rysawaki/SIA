import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


# ============================================================
# 1. Config & Geometry Definition
# ============================================================
class SIAConfig_v6_1:
    d_model = 64  # 世界の次元 (World Dimension)
    trace_dim = 128  # Traceの次元 (Trace Dimension) - d_modelと独立

    # SelfSpace Geometry
    # 自己空間の「広さ（隠れ層サイズ）」
    manifold_dim = 128

    # Metabolism
    metabolic_rate = 0.1  # 空間変容の速度 (0.0=固定, 1.0=即時更新)


# ============================================================
# 2. Trace (Genome)
# ============================================================
class TraceGenome(nn.Module):
    def __init__(self, trace_dim, decay=0.98):
        super().__init__()
        self.register_buffer('genome', torch.zeros(1, trace_dim))
        self.decay = decay

    def imprint(self, signal):
        # signal: [B, trace_dim] -> batch mean
        s = signal.mean(dim=0, keepdim=True)
        self.genome = self.genome * self.decay + s * (1 - self.decay)

    def read(self):
        return self.genome


# ============================================================
# 3. Metabolic Kernel (The Geometric Operator)
#    Trace -> Metric(G) & Center(mu)
# ============================================================
class MetabolicKernel_v6_1(nn.Module):
    """
    Traceから「空間の幾何学（Metric, Center）」を生成するOperator。
    """

    def __init__(self, trace_dim, world_dim, manifold_dim):
        super().__init__()
        self.world_dim = world_dim
        self.manifold_dim = manifold_dim

        # Hypernetwork for Metric (G): world_dim -> manifold_dim
        # Gは「世界をどう歪めて自己空間に射影するか」を決める行列
        self.metric_gen = nn.Sequential(
            nn.Linear(trace_dim, 256),
            nn.SiLU(),
            nn.Linear(256, world_dim * manifold_dim)
        )

        # Hypernetwork for Center (mu): world_dim
        # muは「自己にとっての原点（関心の中心）」
        self.center_gen = nn.Sequential(
            nn.Linear(trace_dim, 128),
            nn.SiLU(),
            nn.Linear(128, world_dim)  # bias term in input space
        )

    def evolve_geometry(self, trace_genome):
        """
        Returns:
            G: [manifold_dim, world_dim] (Metric / Projection)
            mu: [world_dim] (Self Center / Bias)
        """
        # Generate Raw Parameters
        g_flat = self.metric_gen(trace_genome)
        mu_flat = self.center_gen(trace_genome)

        # Reshape
        G = g_flat.view(self.manifold_dim, self.world_dim)
        mu = mu_flat.view(self.world_dim)

        # Scaling for stability (初期の爆発を防ぐ)
        G = G * 0.05

        return G, mu


# ============================================================
# 4. SelfSpace (The Geometric Manifold)
# ============================================================
class GeometricSelfManifold(nn.Module):
    """
    生成された G, mu によって定義される空間。
    y = Activation( G @ (x - mu) )
    幾何学的解釈:
    1. x - mu: 世界を「自己中心」へシフト
    2. G @ ...: 計量テンソルによる歪曲・射影
    3. Activation: 空間の非線形な曲がり
    """

    def __init__(self, config):
        super().__init__()
        self.metabolic_rate = config.metabolic_rate

        # 状態としての現在の幾何学 (Persistent Geometry)
        # 学習パラメータではなく、Metabolismによって更新されるバッファ
        self.register_buffer('current_G', torch.zeros(config.manifold_dim, config.d_model))
        self.register_buffer('current_mu', torch.zeros(config.d_model))

        # 初期化済みフラグ
        self.register_buffer('is_initialized', torch.tensor(0.0))

    def update_geometry(self, new_G, new_mu):
        """
        Metabolism: 空間法則の更新
        急激な変化を防ぐため、metabolic_rate で補間する
        """
        if self.is_initialized == 0.0:
            self.current_G = new_G
            self.current_mu = new_mu
            self.is_initialized.fill_(1.0)
        else:
            eta = self.metabolic_rate
            self.current_G = (1 - eta) * self.current_G + eta * new_G
            self.current_mu = (1 - eta) * self.current_mu + eta * new_mu

    def forward(self, x):
        # x: [B, d_model]

        # 1. Shift to Self-Center (Affine Translation)
        # muは [d_model] なのでbroadcast
        x_centered = x - self.current_mu

        # 2. Apply Metric Distortion (Linear Projection)
        # y = x_centered @ G.T
        # G: [manifold, world], x: [B, world]
        projected = F.linear(x_centered, self.current_G)

        # 3. Manifold Curvature (Nonlinearity)
        # SiLU (Swish) は滑らかな多様体に適している
        out = F.silu(projected)

        return out


# ============================================================
# 5. SIA v6.1 System (Integrated)
# ============================================================
class SIASystem_v6_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Genome (Trace)
        self.trace = TraceGenome(config.trace_dim)
        # Trace Projection: d_model -> trace_dim (次元の独立性を確保)
        self.trace_proj = nn.Linear(config.d_model, config.trace_dim)

        # 2. Operator (Kernel)
        self.kernel = MetabolicKernel_v6_1(
            trace_dim=config.trace_dim,
            world_dim=config.d_model,
            manifold_dim=config.manifold_dim
        )

        # 3. Manifold (Self)
        self.self_manifold = GeometricSelfManifold(config)

        # 4. Perception & Action (Interfaces)
        # MNIST(784) -> World(d_model)
        self.encoder = nn.Linear(784, config.d_model)
        # Manifold(manifold_dim) + World(d_model) -> Action(10)
        self.decoder = nn.Linear(config.d_model + config.manifold_dim, 10)

    def forward(self, x, target=None):
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # --- A. Metabolism (Rule Update) ---
        # 1. Read Trace
        genome = self.trace.read()

        # 2. Evolve Geometry (Generate G, mu)
        new_G, new_mu = self.kernel.evolve_geometry(genome)

        # 3. Update Manifold (Time evolution of space)
        # ここで「自己の法則」が書き換わる
        self.self_manifold.update_geometry(new_G, new_mu)

        # --- B. Interaction (Reality Processing) ---
        # 4. Encode Reality
        world_feat = F.relu(self.encoder(x_flat))  # [B, d_model]

        # 5. Apply Self Manifold
        # 現在の G, mu で世界を歪曲して解釈する
        self_view = self.self_manifold(world_feat)  # [B, manifold_dim]

        # 6. Action / Prediction
        # 残差接続ではなく、Concatenateで「客観 + 主観」を統合
        combined = torch.cat([world_feat, self_view], dim=1)
        logits = self.decoder(combined)

        # --- C. Imprint (Experience Recording) ---
        if target is not None:
            # 7. Calc Pain
            loss = F.cross_entropy(logits, target)

            # 8. Trace Imprint
            # 痛みをTrace次元へ射影して刻む
            pain_signal = self.trace_proj(world_feat.detach()) * loss.item()
            self.trace.imprint(pain_signal)

            return logits, loss

        return logits, 0.0


# ============================================================
# Test Runner
# ============================================================
def test_v6_1():
    config = SIAConfig_v6_1()
    model = SIASystem_v6_1(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy Data
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))

    print("SIA v6.1 Metabolic Space Check")
    print("-" * 50)
    print(f"{'Step':<5} | {'Loss':<8} | {'Genome Δ':<8} | {'|G|':<8} | {'|μ|':<8}")

    initial_genome = model.trace.read().clone()

    for t in range(10):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Check Geometry Evolution
        current_genome = model.trace.read()
        genome_diff = (current_genome - initial_genome).norm().item()

        # 現在の多様体のパラメータを確認
        G_norm = model.self_manifold.current_G.norm().item()
        mu_norm = model.self_manifold.current_mu.norm().item()

        print(f"{t:<5} | {loss.item():.4f}   | {genome_diff:.4f}   | {G_norm:.4f}   | {mu_norm:.4f}")


if __name__ == "__main__":
    test_v6_1()