# muon/projections.py
import torch

# ------------------------------------------------------------
# Original Newton–Schulz (Muon) implementation
# ------------------------------------------------------------
def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate polar factor / orthogonal projection of a 2D gradient matrix via Newton–Schulz.
    Matches your original implementation (coeffs a,b,c).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    X /= (X.norm() + eps)

    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


# ------------------------------------------------------------
# Exact polar projection Q = U V^T (SVD)
# ------------------------------------------------------------
def polar_exact(G: torch.Tensor) -> torch.Tensor:
    """
    Exact polar factor Q = U V^T using thin SVD in float32 (more stable), cast back.
    """
    assert G.ndim == 2
    G32 = G.float()
    U, _, Vh = torch.linalg.svd(G32, full_matrices=False)
    Q = U @ Vh
    return Q.to(dtype=G.dtype, device=G.device)
