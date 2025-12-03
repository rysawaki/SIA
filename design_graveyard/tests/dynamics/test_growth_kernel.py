import torch
from src.identity.core.trace_tensor import TraceTensor
from design_graveyard.growth_kernel import GrowthKernel


def test_continuity_of_update():
    """Update should make a SMALL but non-zero change."""
    dim = 8
    T = TraceTensor.from_dim(dim)

    T_new = GrowthKernel.update(T)

    delta = torch.norm(T_new - T).item()

    # 変化していること（0ではない）
    assert delta > 1e-6, "GrowthKernel did not change the TraceTensor."

    # しかし爆発していないこと（連続性保証）
    assert delta < 1.0, "GrowthKernel update is too large; discontinuity detected."


def test_irreversibility():
    """Update should not be reversible (Memory must accumulate)."""
    T = TraceTensor.from_dim(6)
    T1 = GrowthKernel.update(T)
    T2 = GrowthKernel.update(T1)

    # T2とTは一致してはいけない（不可逆性）
    assert not torch.allclose(T2, T, atol=1e-6), \
        "Irreversibility violated: GrowthKernel should encode accumulating history."


def test_curvature_shift():
    """Eigenvalues should change after update (structure deformation)."""
    dim = 6
    T = TraceTensor.from_dim(dim)

    vals_before, _ = T.eigendecompose()
    T_new = GrowthKernel.update(T)
    vals_after, _ = T_new.eigendecompose()

    # 固有値が完全に一致しているなら、几何学変形が起きていない
    assert not torch.allclose(vals_before, vals_after, atol=1e-6), \
        "Eigenvalues unchanged — Growth is not deforming the tensor structure."


def test_boundedness_of_update():
    """Norm should not explode (stability guarantee)."""
    dim = 8
    T = TraceTensor.from_dim(dim)

    T_new = GrowthKernel.update(T)

    norm_before = torch.norm(T).item()
    norm_after = torch.norm(T_new).item()

    # Norm should remain within a stable bound
    assert norm_after < norm_before * 5, \
        "GrowthKernel update caused norm explosion."

    assert norm_after > norm_before * 0.2, \
        "GrowthKernel update caused excessive decay / collapse."


