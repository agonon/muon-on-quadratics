# muon/data.py
import math
import torch


def _trunc_std_normal(numel: int, k: float, device: str):
    """
    Sample numel values from N(0,1) truncated to [-k, k] via rejection sampling.
    Returns a 1D tensor of length numel on the given device.
    """
    out = torch.empty((numel,), device=device)
    filled = 0
    # For k=3, acceptance ~99.7%, so this is fast.
    while filled < numel:
        need = numel - filled
        z = torch.randn((need,), device=device)
        z = z[z.abs() <= k]
        take = min(z.numel(), need)
        if take > 0:
            out[filled:filled + take] = z[:take]
            filled += take
    return out


def make_s(r, kind="flat", s_max=1.0, s_min=1e-2, alpha=1.0, device="cpu"):
    """
    Returns a length-r vector of eigenvalues in [s_min, s_max],
    sorted descending, with exact endpoints:
      out[0]  == s_max
      out[-1] == s_min

    Kinds:
      - flat_max       (all s_max, except last value s_min)
      - flat_min       (all s_min, except last value s_max)
      - uniform
      - log_uniform
      - geometric
      - geometric_<q>          (q in (0,1))
      - powerlaw               (alpha controls tail)
      - linear_decay_to_smax
      - linear_decay_faster
      - spikes_N_pmin_pmax     (see below)
      - gaussian / gaussian_<k>  (truncated normal on [s_min,s_max], mean mid, trunc at ±k std; default k=3)
      - u_shaped / u_shaped_<a> / u_shaped_{weak,medium,strong}
           symmetric Beta(a,a) mapped to [s_min,s_max], with 0<a<1 for U-shape

    Notes:
      - Endpoints are forced: s[0]=s_max, s[-1]=s_min after sampling and sorting.
      - gaussian uses true truncation for exact support [s_min,s_max].
      - u_shaped uses Beta(a,a) on [0,1] then affine map to [s_min,s_max].
    """
    s_min = float(s_min)
    s_max = float(s_max)
    assert s_min > 0.0 and s_max >= s_min

    if kind == "flat_max":
        s = torch.full((r,), s_max, device=device)
        s[-1] = s_min

    elif kind == "flat_min":
        s = torch.full((r,), s_min, device=device)
        s[-1] = s_max

    elif kind == "uniform":
        s = s_min + (s_max - s_min) * torch.rand(r, device=device)

    elif kind == "log_uniform":
        u = torch.rand(r, device=device)
        s = s_min * (s_max / s_min) ** u

    elif kind == "geometric":
        s = torch.logspace(math.log10(s_max), math.log10(s_min), r, device=device)

    elif kind == "powerlaw":
        i = torch.arange(1, r + 1, device=device, dtype=torch.float32)
        raw = i.pow(-alpha)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)  # in [0,1]
        s = s_min + (s_max - s_min) * (1.0 - raw)

    elif kind == "linear_decay_to_smax":
        u = torch.rand(r, device=device)
        s = s_max - (s_max - s_min) * torch.sqrt(u)

    elif kind == "linear_decay_faster":
        u = torch.rand(r, device=device)
        s = s_min + 2 - 2 * torch.sqrt(u)

    elif kind == "gaussian" or kind.startswith("gaussian_"):
        # Truncated normal on [s_min, s_max] with mean at mid and trunc at ±k std.
        # Choose sigma so that mid ± k*sigma hits endpoints exactly.
        if kind == "gaussian":
            k = 3.0
        else:
            try:
                k = float(kind.split("gaussian_", 1)[1])
            except Exception as e:
                raise ValueError(f"Could not parse k from kind='{kind}'. Expected 'gaussian_<k>' like gaussian_2.5") from e
        if k <= 0:
            raise ValueError(f"gaussian k must be > 0, got {k}")

        mid = 0.5 * (s_min + s_max)
        sigma = (s_max - s_min) / (2.0 * k)

        z = _trunc_std_normal(r, k=k, device=device)  # in [-k, k]
        s = mid + sigma * z
        s = torch.clamp(s, min=s_min, max=s_max)

    elif kind == "u_shaped" or kind.startswith("u_shaped_"):
        # Supports:
        #   - "u_shaped"         -> medium (a=0.5)
        #   - "u_shaped_weak"    -> mild U-shape (a=0.9)
        #   - "u_shaped_medium"  -> medium (a=0.5)
        #   - "u_shaped_strong"  -> strong U-shape (a=0.2)
        #   - "u_shaped_<a>"     -> user-specified a in (0,1), e.g. u_shaped_0.95 or u_shaped_0.1
        if kind == "u_shaped":
            a = 0.5
        else:
            tag = kind.split("u_shaped_", 1)[1]
            presets = {"weak": 0.9, "medium": 0.5, "strong": 0.2}
            if tag in presets:
                a = presets[tag]
            else:
                try:
                    a = float(tag)
                except Exception as e:
                    raise ValueError(
                        f"Could not parse kind='{kind}'. Use 'u_shaped', "
                        f"'u_shaped_weak|medium|strong', or 'u_shaped_<a>' like u_shaped_0.8."
                    ) from e

        if not (0.0 < a < 1.0):
            raise ValueError(f"u_shaped requires 0 < a < 1 for a U-shape; got a={a}")

        u = torch.distributions.Beta(a, a).sample((r,)).to(device)
        s = s_min + (s_max - s_min) * u

    # expected: "geometric_<q>"
    elif "geometric_" in kind:
        try:
            q = float(kind.split("geometric_", 1)[1])
        except Exception as e:
            raise ValueError(f"Could not parse q from kind='{kind}'. Expected like 'geometric_0.93'.") from e
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0,1); got q={q} from kind='{kind}'")

        i = torch.arange(r, device=device, dtype=torch.float32)
        s = s_max * (q ** i)
        s = torch.clamp(s, min=s_min)

    # kind format: "spikes_N_pmin_pmax" e.g. "spikes_6_0.2_0.05"
    elif "spikes_" in kind:
        try:
            _, N_str, pmin_str, pmax_str = kind.split("_", 3)
            N = int(N_str)
            pmin = float(pmin_str)
            pmax = float(pmax_str)
        except Exception as e:
            raise ValueError(
                f"Could not parse kind='{kind}'. Expected 'spikes_N_pmin_pmax' like 'spikes_6_0.2_0.05'."
            ) from e

        if N < 2:
            raise ValueError(f"N must be >= 2, got N={N} from kind='{kind}'")
        if not (0.0 <= pmin <= 1.0 and 0.0 <= pmax <= 1.0):
            raise ValueError(f"pmin,pmax must be in [0,1], got pmin={pmin}, pmax={pmax}")
        if pmin + pmax > 1.0:
            raise ValueError(f"Need pmin+pmax <= 1, got {pmin+pmax}")

        n_min = max(1, int(round(pmin * r)))
        n_max = max(1, int(round(pmax * r)))

        while n_min + n_max > r:
            if n_min >= n_max and n_min > 1:
                n_min -= 1
            elif n_max > 1:
                n_max -= 1
            else:
                break

        rem = r - (n_min + n_max)

        if N == 2:
            if rem > 0:
                k = int(torch.randint(low=0, high=rem + 1, size=(1,), device=device).item())
                n_min += k
                n_max += rem - k
            s = torch.cat([
                torch.full((n_max,), s_max, device=device),
                torch.full((n_min,), s_min, device=device),
            ])
        else:
            mid_vals = s_min + (s_max - s_min) * torch.rand(N - 2, device=device)

            if rem > 0:
                probs = torch.full((N - 2,), 1.0 / (N - 2), device=device)
                mid_counts = torch.multinomial(probs, num_samples=rem, replacement=True)
                counts = torch.bincount(mid_counts, minlength=N - 2)
            else:
                counts = torch.zeros(N - 2, device=device, dtype=torch.long)

            parts = [
                torch.full((n_max,), s_max, device=device),
                torch.full((n_min,), s_min, device=device),
            ]
            for v, c in zip(mid_vals, counts.tolist()):
                if c > 0:
                    parts.append(torch.full((c,), v.item(), device=device))
            s = torch.cat(parts)

        if s.numel() != r:
            raise RuntimeError(f"Internal error: built s has length {s.numel()} != r={r}")
        s = torch.sort(s, descending=True).values

    else:
        raise ValueError(f"unknown kind {kind}")

    s = torch.sort(s, descending=True).values
    s[0] = s_max
    s[-1] = s_min
    return s


def make_X_from_singular_values(n, d_in, s, device, dtype=torch.float32):
    """
    X = U diag(s) V^T with U in R^{n×r}, V in R^{d_in×r} having orthonormal columns.
    For full-rank A=X^T X, you want n >= d_in and r=d_in with all s>0.
    """
    r = s.numel()
    U, _ = torch.linalg.qr(torch.randn(n, r, device=device, dtype=dtype), mode="reduced")
    V, _ = torch.linalg.qr(torch.randn(d_in, r, device=device, dtype=dtype), mode="reduced")
    return (U * s) @ V.T


def build_instance(kind, s_min, s_max, n, d_in, d_out, device, W_star, seed=0, alpha=1.0):
    """
    Specify s_min/s_max as the desired min/max eigenvalues of A, where
      A = (X^T X) / scale,  scale = n * d_out.

    We sample the remaining eigenvalues according to 'kind' via make_s(...),
    then set singular values of X as sigma_i = sqrt(scale * eig_i).
    """
    torch.manual_seed(seed)
    assert n >= d_in, "Need n >= d_in for A=X^T X to be full rank without ridge."
    r = d_in
    scale = n * d_out

    # eigenvalues of A in [s_min, s_max]
    eigs = make_s(r, kind=kind, s_max=s_max, s_min=s_min, alpha=alpha, device=device)

    # singular values of X so that eig(A)=sigma^2/scale
    sigmas = torch.sqrt(eigs * scale)

    X = make_X_from_singular_values(n, d_in, sigmas, device=device)
    Y = X @ W_star

    A = (X.T @ X) / scale
    B = -(X.T @ Y) / scale
    c = 0.5 * (Y * Y).sum() / scale

    def loss_fn(params):
        (W,) = params
        return 0.5 * (W * (A @ W)).sum() + (W * B).sum() + c

    with torch.no_grad():
        evals = torch.linalg.eigvalsh(0.5 * (A + A.T))
        kappa_actual = float((evals.max() / evals.min()).item())

    return loss_fn, X, Y, A, kappa_actual, evals
