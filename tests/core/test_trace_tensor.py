import torch
from identity.core.trace_tensor import TraceTensor


def test_trace_tensor_basic_properties():
    dim = 8
    T = TraceTensor.from_dim(dim)

    # Symmetry test
    assert torch.allclose(T.T, T.T.T, atol=1e-6), "TraceTensor must be symmetric"

    # Eigenvalues should be real
    vals, vecs = T.eigendecompose()
    assert torch.all(vals.imag.abs() < 1e-6), "Eigenvalues must be real"


def test_rank1_update_increases_anisotropy():
    dim = 8
    T = TraceTensor.from_dim(dim)
    u = torch.randn(dim)

    before = T.anisotropy().item()
    T.add_rank1_(u, strength=5.0)
    after = T.anisotropy().item()
    assert after > before, "Rank-1 update should increase anisotropy"


def test_normalization_prevents_explosion():
    dim = 8
    T = TraceTensor.from_dim(dim)

    for _ in range(10):
        u = torch.randn(dim)
        T.add_rank1_(u, strength=10.0)

    T.normalize_spectral_(max_eig=1.0)
    vals, _ = T.eigendecompose()

    assert torch.max(torch.abs(vals)) <= 1.01, "Normalization failed to control eigenvalues"


def test_sectional_curvature_basic():
    dim = 8
    T = TraceTensor.from_dim(dim, init_scale=0.1)

    u = torch.randn(dim)
    v = torch.randn(dim)

    K = T.sectional_curvature(u, v)
    assert K.dim() == 0, "Sectional curvature must be scalar"
