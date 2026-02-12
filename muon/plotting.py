# muon/plotting.py
import os
import re
import math
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Pretty display names + optional filtering (set to None to drop)
KIND_RENAME = {
    "flat_max": "max_spiked",
    "flat_min": "min_spiked",
    "gaussian": "gaussian",
    "geometric_0.9": "geometric_decay_to_max",
    "linear_decay_to_smax": "linear_decay_to_max",
    "linear_decay_faster": "linear_decay_to_max",
    "u_shaped_strong": "u_shaped",
    "u_shaped_weak": None,   # drop
    "uniform": "uniform",
}


# Color-by-family, shade-by-lr helpers
_LR_RE = re.compile(r"(?:^|[_-])lr[_=]?([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.IGNORECASE)


def _parse_lr(name: str) -> Optional[float]:
    """
    Extract lr from algorithm name.
    Supports:
      - '..._lr_0.01'
      - '..._lr0.01'
      - 'GD_lr_1e-3'
    Returns float or None.
    """
    m = _LR_RE.search(name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m2 = re.search(r"lr([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", name, re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None

    return None


def _family_key(name: str) -> str:
    """
    Remove lr token from name to define an 'algorithm family'.
    Keeps projection/momentum/nesterov tokens, etc., so these remain separate families.
    """
    s = re.sub(r"([_-])lr[_=]?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?", "", name, flags=re.IGNORECASE)
    s = re.sub(r"__+", "_", s).strip("_-")
    return s


def _mix_with_white(rgb, t: float):
    """
    Mix color with white by fraction t in [0,1].
      t=0 -> original color (dark)
      t=1 -> white (light)
    """
    r, g, b = rgb
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _shade_for_lr(base_color, lr: Optional[float], lr_list,
                  light_span: float = 0.75,
                  darken_max: float = 0.18):
    """
    Small lr -> lighter (mix with white).
    Largest lr -> slightly darker than base_color.
    """
    base_rgb = np.array(mcolors.to_rgb(base_color), dtype=float)

    if lr is None or not lr_list:
        return tuple(base_rgb)

    uniq = sorted(set(lr_list))
    if len(uniq) == 1:
        rank = 1.0
    else:
        rank = uniq.index(lr) / (len(uniq) - 1)  # 0 small -> 1 large

    t_light = light_span * (1.0 - rank) 
    rgb = np.array(_mix_with_white(tuple(base_rgb), t_light), dtype=float)

    # extra darkening only for large ranks:
    # rank=0 -> no darkening, rank=1 -> darken_max
    d = darken_max * (rank ** 2)
    rgb = rgb * (1.0 - d)

    # clip for safety
    rgb = np.clip(rgb, 0.0, 1.0)
    return tuple(rgb)


# utility helpers for drilldowns
def _sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("._-")


def _group_by_family(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for name, tr in results.items():
        fam = _family_key(name)
        groups.setdefault(fam, []).append((name, tr))
    return groups


def _score_trace_tail(tr: Dict[str, Any], metric: str = "loss", tail: int = 25, reducer: str = "mean") -> float:
    x = tr.get(metric, None)
    if x is None:
        return float("inf")

    # support torch tensors, lists, numpy arrays
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=float)

    if tail is not None and tail > 0 and len(x) > tail:
        x = x[-tail:]

    if reducer == "median":
        return float(np.median(x))
    return float(np.mean(x))


def _pick_best_variant(
    items: List[Tuple[str, Dict[str, Any]]],
    metric: str = "loss",
    tail: int = 25,
    reducer: str = "mean",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not items:
        return None, None
    best_name, best_tr = min(items, key=lambda nt: _score_trace_tail(nt[1], metric=metric, tail=tail, reducer=reducer))
    return best_name, best_tr


def _select_baseline(
    results: Dict[str, Dict[str, Any]],
    family_pattern: str,
    metric: str = "loss",
    tail: int = 25,
    reducer: str = "mean",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Select "best" run among all names whose FAMILY matches family_pattern (regex on _family_key(name)).
    """
    pat = re.compile(family_pattern, re.IGNORECASE)
    items = [(nm, tr) for nm, tr in results.items() if pat.search(_family_key(nm))]
    return _pick_best_variant(items, metric=metric, tail=tail, reducer=reducer)


# Main plotting function
def plot_traces_side_by_side(
    results: Dict[str, Dict[str, Any]],
    n: int,
    d_in: int,
    d_out: int,
    steps: int,
    A: torch.Tensor,
    show_grad_norm: bool = True,
    show_grad_cond_num: bool = True,
    show_eig_hist: bool = True,
    title_prefix: str = "",
    savepath: Optional[str] = None,
    show: bool = True,
    style_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    label_overrides: Optional[Dict[str, str]] = None,
):
    """
    Plots one figure per experiment:
      - loss (always)
      - grad norm (optional)
      - grad condition number (optional)
      - eigenvalue histogram (always)
      - err_F / err_C (optional if present)

    Styling:
      - one base color per algorithm family (name with lr removed)
      - lr within a family shown as lighter/darker shade (darker = larger lr)
      - style_overrides: per-name kwargs passed into ax.plot (e.g. color='k', linestyle='--', linewidth=2.5)
      - label_overrides: per-name label string
    """
    LOG_CUTOFF = 1e-5
    style_overrides = style_overrides or {}
    label_overrides = label_overrides or {}

    first = next(iter(results.values()))
    sched = first.get("sched", "None")

    ev_min = ev_max = kappa = None
    evals = None
    if show_eig_hist:
        # eigenvalues of A (symmetrize defensively)
        with torch.no_grad():
            A_sym = 0.5 * (A + A.T)
            evals = torch.linalg.eigvalsh(A_sym).detach().cpu()
            ev_min = float(evals.min().item())
            ev_max = float(evals.max().item())
            kappa = (ev_max / ev_min) if ev_min > 0 else float("inf")

    # detect whether we can plot subspace errors
    has_subspace = any(("err_F" in tr and "err_C" in tr) for tr in results.values())

    # Build axes list in the order we want
    axes_specs = [("loss", None)]
    if show_grad_norm:
        axes_specs.append(("grad_norm", None))
    if show_grad_cond_num:
        axes_specs.append(("grad_cond_num", None))
    if show_eig_hist:
        axes_specs.append(("eig_hist", None))
    if has_subspace:
        axes_specs.append(("err_F", None))
        axes_specs.append(("err_C", None))

    ncols = len(axes_specs)
    fig, axes = plt.subplots(1, ncols, figsize=(10.0 * ncols, 7), dpi=140)
    if ncols == 1:
        axes = [axes]

    ax = {name: axes[i] for i, (name, _) in enumerate(axes_specs)}

    # Color mapping: family -> base, and shade by lr
    family_palette = list(plt.get_cmap("tab10").colors)

    names = list(results.keys())
    families = [_family_key(nm) for nm in names]
    uniq_families = sorted(set(families))
    fam2color = {fam: family_palette[i % len(family_palette)] for i, fam in enumerate(uniq_families)}

    fam2lrs = {}
    for nm in names:
        fam = _family_key(nm)
        lr = _parse_lr(nm)
        fam2lrs.setdefault(fam, [])
        if lr is not None:
            fam2lrs[fam].append(lr)

    def color_for_name(nm: str):
        # allow style override to force a color
        if nm in style_overrides and "color" in style_overrides[nm]:
            return style_overrides[nm]["color"]
        fam = _family_key(nm)
        lr = _parse_lr(nm)
        return _shade_for_lr(fam2color[fam], lr, fam2lrs.get(fam, []))

    def label_for_name(nm: str):
        if nm in label_overrides:
            return label_overrides[nm]
        fam = _family_key(nm)
        lr = _parse_lr(nm)
        if lr is None:
            return fam
        return f"{fam} | lr={lr:g}"

    def _prep_for_log(y, cutoff=LOG_CUTOFF):
        # y can be torch/list/np; return 1D numpy
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        y = np.asarray(y, dtype=float).reshape(-1)

        # replace non-finite with nan so matplotlib breaks the line
        y[~np.isfinite(y)] = np.nan

        # clamp to cutoff for log visibility
        y = np.maximum(y, cutoff)
        return y

    def _plot_series(ax_here, series, nm: str):
        base_kwargs = dict(
            label=label_for_name(nm),
            color=color_for_name(nm),
            alpha=0.95,
        )
        # let user override any plot kwargs
        base_kwargs.update(style_overrides.get(nm, {}))

        y = _prep_for_log(series)
        ax_here.plot(y, **base_kwargs)


    # ---- Loss (always) ----
    axL = ax["loss"]
    for name, tr in results.items():
        _plot_series(axL, tr["loss"], name)
    axL.set_yscale("log")
    axL.set_xlabel("step")
    axL.set_ylabel("loss (log)")
    axL.grid(True, which="both", alpha=0.25)

    # ---- Grad norm (optional) ----
    if "grad_norm" in ax:
        axG = ax["grad_norm"]
        for name, tr in results.items():
            if "grad_norm" in tr:
                _plot_series(axG, tr["grad_norm"], name)
        axG.set_yscale("log")
        axG.set_xlabel("step")
        axG.set_ylabel(r"$\|\nabla\|_F$ (log)")
        axG.grid(True, which="both", alpha=0.25)

    # ---- Grad condition number (optional) ----
    if "grad_cond_num" in ax:
        axC = ax["grad_cond_num"]
        for name, tr in results.items():
            if "grad_cond_num" in tr:
                _plot_series(axC, tr["grad_cond_num"], name)
        axC.set_yscale("log")
        axC.set_xlabel("step")
        axC.set_ylabel("cond(grad) (log)")
        axC.grid(True, which="both", alpha=0.25)

    # ---- Eigenvalue histogram (optional) ----
    if show_eig_hist:
        axE = ax["eig_hist"]
        axE.hist(evals.numpy(), bins=40)
        axE.set_xlabel("eigenvalue")
        axE.set_ylabel("count")
        axE.set_title(f"eig(A): min={ev_min:.2e}, max={ev_max:.2e}, κ≈{kappa:.2e}")
        axE.grid(True, alpha=0.25)

    # ---- Subspace errors (optional, auto) ----
    if has_subspace:
        axF = ax["err_F"]
        axCuv = ax["err_C"]
        for name, tr in results.items():
            if ("err_F" not in tr) or ("err_C" not in tr):
                continue
            _plot_series(axF, tr["err_F"], name)
            _plot_series(axCuv, tr["err_C"], name)

        axF.set_yscale("log")
        axF.set_xlabel("step")
        axF.set_ylabel("flat subspace error (log)")
        axF.grid(True, which="both", alpha=0.25)

        axCuv.set_yscale("log")
        axCuv.set_xlabel("step")
        axCuv.set_ylabel("curved subspace error (log)")
        axCuv.grid(True, which="both", alpha=0.25)

    # Legend below (shared) — pull from loss axis
    handles, labels = axL.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.legend(handles, labels, loc="upper right", frameon=False,  bbox_to_anchor=(1.0, 0.97), fontsize=9)
    plt.tight_layout()

    if savepath is not None:
        print("SAVING TO: ", savepath)
        d = os.path.dirname(savepath)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(savepath, dpi=200, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

def extract_kind(s: str) -> str:
    """
    Extract kind from strings like:
      "kind=flat_max_n=128_din=... "
    Underscores are allowed inside kind.
    """
    m = re.search(r"kind=(.*?)_n=", s)
    if not m:
        raise ValueError(f"Could not parse kind from: {s}")
    return m.group(1)


def plot_family_drilldowns(
    results: Dict[str, Dict[str, Any]],
    n: int,
    d_in: int,
    d_out: int,
    steps: int,
    A: torch.Tensor,
    title_prefix: str = "",
    out_dir: Optional[str] = None,
    show: bool = False,
    baseline_patterns: Optional[Dict[str, str]] = None,
    baseline_metric: str = "loss",
    baseline_tail: int = 25,
    baseline_reducer: str = "mean",
    include_baselines_on_own_family: bool = False,
    show_grad_norm: bool = True,
    show_grad_cond_num: bool = True,
    show_eig_hist: bool = False,
    focus_base_color: str = "tab:blue",
    baseline_color_map: Optional[Dict[str, str]] = None,
):
    """
    One plot per family: show only that family's lr variants (shades of focus_base_color),
    plus global baselines (distinct fixed colors).
    """
    if baseline_patterns is None:
        baseline_patterns = {"GD": r"^GD$", "Adam": r"^Adam$"}

    if baseline_color_map is None:
        baseline_color_map = {
            "GD": "tab:orange",
            "Adam": "tab:green",
        }

    groups = _group_by_family(results)

    # ---- select baselines globally (best run matching the pattern) ----
    baselines: Dict[str, Dict[str, Any]] = {}
    baseline_meta: Dict[str, Tuple[str, Optional[float]]] = {}

    for disp, pat in baseline_patterns.items():
        nm, tr = _select_baseline(
            results,
            family_pattern=pat,
            metric=baseline_metric,
            tail=baseline_tail,
            reducer=baseline_reducer,
        )
        if nm is None or tr is None:
            continue
        key = f"__BASELINE__{disp}"
        baselines[key] = tr
        baseline_meta[disp] = (nm, _parse_lr(nm))

    # baseline styling: distinct colors, thick dashed, on top
    baseline_style: Dict[str, Dict[str, Any]] = {}
    baseline_label: Dict[str, str] = {}
    for disp in baseline_patterns.keys():
        key = f"__BASELINE__{disp}"
        if key not in baselines:
            continue
        orig_nm, lr = baseline_meta.get(disp, ("", None))
        lr_txt = f"{lr:g}" if lr is not None else "?"
        baseline_label[key] = f"{disp} (best lr={lr_txt})"
        baseline_style[key] = dict(
            color=baseline_color_map.get(disp, "k"),
            linewidth=2,
            alpha=0.9,
            zorder=10,
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for fam, items in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        # sub contains ONLY the focus family variants to start
        sub = {nm: tr for nm, tr in items}

        # focus family styling: one hue, shade by lr
        focus_names = [nm for nm, _ in items]
        focus_lrs = [lr for lr in (_parse_lr(nm) for nm in focus_names) if lr is not None]

        focus_style: Dict[str, Dict[str, Any]] = {}
        for nm in focus_names:
            lr = _parse_lr(nm)
            focus_style[nm] = dict(
                color=_shade_for_lr(focus_base_color, lr, focus_lrs),
                alpha=0.7,
                linewidth=1.8,
                zorder=3,
            )

        # add baselines (optionally skip on their own family plot)
        if baselines:
            for bkey, btr in baselines.items():
                if not include_baselines_on_own_family:
                    disp = bkey.replace("__BASELINE__", "")
                    pat = baseline_patterns.get(disp, None)
                    if pat is not None and re.search(pat, fam, flags=re.IGNORECASE):
                        continue
                sub[bkey] = btr

        # merge styles (baseline overrides win if key collision, but there won't be)
        style_overrides = {}
        style_overrides.update(focus_style)
        style_overrides.update(baseline_style)

        label_overrides = dict(baseline_label)

        savepath = None
        if out_dir is not None:
            savepath = os.path.join(out_dir, f"{_sanitize_filename(fam)}.pdf")

        plot_traces_side_by_side(
            sub,
            n=n, d_in=d_in, d_out=d_out,
            steps=steps,
            A=A,
            show_grad_norm=show_grad_norm,
            show_grad_cond_num=show_grad_cond_num,
            show_eig_hist=show_eig_hist,
            title_prefix=f"{title_prefix} | family={fam}",
            savepath=savepath,
            show=show,
            style_overrides=style_overrides,
            label_overrides=label_overrides,
        )


    def _parse_mom(fam: str):
        m = re.search(r"mom([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", fam)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _is_exact(fam: str) -> bool:
        return fam.lower().startswith("muon_exact")

    def _is_ns(fam: str) -> bool:
        return fam.lower().startswith("muon_ns")

    # collect keys per category (by family, not by raw key)
    cats = [
        ("Exact projection, no momentum",  "exact_nomom"),
        ("Exact projection, momentum",     "exact_mom"),
        ("Newton-Schulz projection, no momentum",     "ns_nomom"),
        ("Newton-Schulz projection, momentum",        "ns_mom"),
    ]
    keys_by_cat = {c: [] for _, c in cats}

    for nm in results.keys():
        fam = _family_key(nm)
        if not fam.lower().startswith("muon"):
            continue
        mom = _parse_mom(fam)
        has_mom = (mom is not None) and (abs(mom) > 1e-15)

        if _is_exact(fam) and (not has_mom):
            keys_by_cat["exact_nomom"].append(nm)
        elif _is_exact(fam) and has_mom:
            keys_by_cat["exact_mom"].append(nm)
        elif _is_ns(fam) and (not has_mom):
            keys_by_cat["ns_nomom"].append(nm)
        elif _is_ns(fam) and has_mom:
            keys_by_cat["ns_mom"].append(nm)

    base_colors = {
        "exact_nomom": "tab:blue",
        "exact_mom":   "#4B0082",
        "ns_nomom":    "tab:green",
        "ns_mom":      "tab:red",
    }

    fig, axs = plt.subplots(2, 2, figsize=(7.0 * 2, 5.2 * 2), dpi=140, sharey=True)
    axs = axs.ravel()

    x = np.arange(steps)

    for j, (panel_title, cat) in enumerate(cats):
        axh = axs[j]
        keys = keys_by_cat[cat]

        axh.set_title(panel_title)
        axh.set_xlabel("step")
        if j % 2 == 0:  # left column
            axh.set_ylabel("loss (log)")
        axh.set_yscale("log")
        axh.grid(True, which="both", alpha=0.25)

        if not keys:
            axh.text(0.5, 0.5, "no runs found", ha="center", va="center", transform=axh.transAxes)
            continue

        # sort by lr, shade within this panel
        keys = sorted(keys, key=lambda k: (_parse_lr(k) is None, _parse_lr(k) or 0.0))
        lrs = [lr for lr in (_parse_lr(k) for k in keys) if lr is not None]

        for nm in keys:
            tr = results.get(nm, None)
            if tr is None or "loss" not in tr:
                continue

            y = tr["loss"]
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            y = np.asarray(y, dtype=float).reshape(-1)[:steps]

            lr = _parse_lr(nm)
            col = _shade_for_lr(base_colors[cat], lr, lrs)
            lab = f"lr={lr:g}" if lr is not None else _family_key(nm)

            axh.plot(x[:len(y)], y, color=col, alpha=0.6, linewidth=1.8, label=lab)

        axh.legend(frameon=False, fontsize=8)

    kind = extract_kind(title_prefix)

    fig.tight_layout()

    if out_dir is not None:
        muon4_path = os.path.join(out_dir, f"{kind}_all_muon_variants.pdf")
    fig.savefig(muon4_path, dpi=200, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] wrote muon variants: {muon4_path}")


# Mean + CI trajectory plotting (across exp seeds)
def _to_1d_np(x, steps: int) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float)

    # squeeze singleton dims, but keep time axis
    x = np.squeeze(x)

    # If still not 1D, assume time is axis 0 and reduce the rest.
    # (This prevents accidental flattening that can interleave columns.)
    if x.ndim > 1:
        # common cases: (T,1), (T,k)
        x = x[:, 0]

    x = x.astype(float).reshape(-1)

    if x.shape[0] < steps:
        if x.shape[0] == 0:
            x = np.full((steps,), np.nan, dtype=float)
        else:
            x = np.pad(x, (0, steps - x.shape[0]), mode="edge")
    else:
        x = x[:steps]
    return x



def _stack_metric(results_list: List[Dict[str, Dict[str, Any]]], key: str, metric: str, steps: int) -> np.ndarray:
    """
    Stack metric over experiments. Returns [K, steps] (K = #exps where present).
    """
    rows = []
    for res in results_list:
        tr = res.get(key, None)
        if tr is None:
            continue
        if metric not in tr:
            continue
        rows.append(_to_1d_np(tr[metric], steps))
    if not rows:
        return np.zeros((0, steps), dtype=float)
    res = np.stack(rows, axis=0)

    return res


def _mean_ci_band(Y: np.ndarray, ci_mult: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y: [K, T] -> mean, lo, hi using mean ± ci_mult * SE, NaN-safe.
    """
    # print('Y.shape: ', Y.shape)

    if Y.shape[0] == 0:
        T = Y.shape[1]
        nan = np.full((T,), np.nan)
        return nan, nan, nan

    mean = np.nanmean(Y, axis=0)
    K_eff = np.sum(np.isfinite(Y), axis=0)  # effective n per time step

    # sample std with ddof=1 where possible
    std = np.full_like(mean, np.nan)
    for t in range(Y.shape[1]):
        yt = Y[:, t]
        yt = yt[np.isfinite(yt)]
        if yt.size >= 2:
            std[t] = yt.std(ddof=1)
        elif yt.size == 1:
            std[t] = 0.0

    se = std / np.sqrt(np.maximum(K_eff, 1))

    lo = mean - ci_mult * se
    hi = mean + ci_mult * se
    return mean, lo, hi


def _mean_ci_band_log(Y: np.ndarray, ci_mult: float = 1.96, eps: float = 1e-300):
    """
    Compute mean and CI in log space:
      z = log(y + eps)
      mean_z ± ci_mult * SE_z
    Then return exp(...) back on original scale.

    Returns (mean, lo, hi) on ORIGINAL scale.
    Interpretation: geometric mean with multiplicative CI.
    """
    if Y.shape[0] == 0:
        T = Y.shape[1]
        nan = np.full((T,), np.nan)
        return nan, nan, nan

    Y = np.asarray(Y, dtype=float)

    # Mask invalid/ nonpositive before logging
    Y = Y.copy()
    Y[~np.isfinite(Y)] = np.nan
    Y[Y <= 0] = np.nan

    Z = np.log(Y + eps)  # eps prevents log(0) if Y is tiny positive (won't rescue NaNs)

    mean_z = np.nanmean(Z, axis=0)

    # effective counts per time step
    n_eff = np.sum(np.isfinite(Z), axis=0)

    # sample std in log space (ddof=1 where possible)
    std_z = np.full_like(mean_z, np.nan)
    for t in range(Z.shape[1]):
        zt = Z[:, t]
        zt = zt[np.isfinite(zt)]
        if zt.size >= 2:
            std_z[t] = zt.std(ddof=1)
        elif zt.size == 1:
            std_z[t] = 0.0

    se_z = std_z / np.sqrt(np.maximum(n_eff, 1))

    lo_z = mean_z - ci_mult * se_z
    hi_z = mean_z + ci_mult * se_z

    mean = np.exp(mean_z)
    lo   = np.exp(lo_z)
    hi   = np.exp(hi_z)

    # if mean_z was nan, exp(nan)=nan already; keep as-is
    return mean, lo, hi


def _best_key_by_final_mean(
    results_list: List[Dict[str, Dict[str, Any]]],
    keys: List[str],
    steps: int,
    metric: str = "loss",
    tail: int = 1,   # tail=1 => last iterate, else average over last tail steps
) -> Optional[str]:
    best_k = None
    best_val = float("inf")
    for k in keys:
        Y = _stack_metric(results_list, k, metric, steps)
        if Y.shape[0] == 0:
            continue
        m = Y.mean(axis=0)
        val = float(m[-1]) if tail <= 1 else float(m[-tail:].mean())
        if val < best_val:
            best_val = val
            best_k = k
    return best_k


def _select_muon_family(
    results_list: List[Dict[str, Dict[str, Any]]],
    steps: int,
    muon_family: Optional[str] = None,              # exact family name after stripping lr
    muon_family_pattern: Optional[str] = None,      # regex on family name
    metric_for_choice: str = "loss",
) -> Optional[str]:
    """
    Choose one Muon family:
      - if muon_family is provided: use it
      - elif muon_family_pattern is provided: first matching family (sorted)
      - else: auto-pick the Muon_* family whose BEST lr has smallest final mean loss
    """
    # collect keys/families
    all_keys = sorted({k for res in results_list for k in res.keys()})
    families = sorted({_family_key(k) for k in all_keys})

    if muon_family is not None:
        return muon_family

    if muon_family_pattern is not None:
        pat = re.compile(muon_family_pattern, re.IGNORECASE)
        cands = [f for f in families if pat.search(f)]
        return cands[0] if cands else None

    # auto: any family starting with "Muon"
    mu_fams = [f for f in families if f.lower().startswith("muon")]
    if not mu_fams:
        return None

    # score each family by: min over its lrs of final mean(metric)
    best_fam = None
    best_score = float("inf")
    for fam in mu_fams:
        fam_keys = [k for k in all_keys if _family_key(k) == fam]
        if not fam_keys:
            continue
        best_key = _best_key_by_final_mean(results_list, fam_keys, steps, metric=metric_for_choice, tail=1)
        if best_key is None:
            continue
        Y = _stack_metric(results_list, best_key, metric_for_choice, steps)
        if Y.shape[0] == 0:
            continue
        score = float(Y.mean(axis=0)[-1])
        if score < best_score:
            best_score = score
            best_fam = fam

    return best_fam


def _central_quantiles_from_z(ci_mult: float) -> Tuple[float, float, float]:
    """
    Map a z-multiplier (like 1.96) to central probability mass under N(0,1),
    then return (q_lo, q_mid, q_hi) for a central quantile band.

    P(|Z| <= z) = erf(z / sqrt(2)).
    So central mass = erf(z / sqrt(2)).
    q_lo = (1 - mass)/2, q_hi = 1 - q_lo, q_mid = 0.5

    For z=1.96 -> mass≈0.95 -> q_lo≈0.025, q_hi≈0.975
    """
    z = float(ci_mult)
    if not math.isfinite(z) or z <= 0:
        # default to IQR if nonsense
        return 0.25, 0.5, 0.75

    mass = math.erf(z / math.sqrt(2.0))  # in (0,1)
    mass = min(max(mass, 0.0), 1.0)
    q_lo = 0.5 * (1.0 - mass)
    q_hi = 1.0 - q_lo
    # numeric safety
    q_lo = min(max(q_lo, 0.0), 0.5)
    q_hi = min(max(q_hi, 0.5), 1.0)

    print('Using q_lo, q_hi = ', q_lo, " ", q_hi)
    return q_lo, 0.5, q_hi


def _quantile_band(Y: np.ndarray, q_lo: float, q_mid: float, q_hi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y: [K,T] -> (mid, lo, hi) via NaN-safe quantiles along axis=0.
    """
    if Y.shape[0] == 0:
        T = Y.shape[1]
        nan = np.full((T,), np.nan)
        return nan, nan, nan

    Y = np.asarray(Y, dtype=float)
    Y = Y.copy()
    Y[~np.isfinite(Y)] = np.nan

    # np.nanquantile handles NaNs, returning NaN if all are NaN at a time step
    mid = np.nanquantile(Y, q_mid, axis=0)
    lo  = np.nanquantile(Y, q_lo,  axis=0)
    hi  = np.nanquantile(Y, q_hi,  axis=0)
    return mid, lo, hi


def _best_key_by_final_median(
    results_list: List[Dict[str, Dict[str, Any]]],
    keys: List[str],
    steps: int,
    metric: str = "loss",
    tail: int = 1,
) -> Optional[str]:
    """
    Pick best key by minimizing the median across experiments of:
      - last iterate if tail<=1
      - mean over last 'tail' iterates (per experiment) if tail>1
    """
    best_k = None
    best_val = float("inf")

    for k in keys:
        Y = _stack_metric(results_list, k, metric, steps)  # [K,T]
        if Y.shape[0] == 0:
            continue

        if tail is None or tail <= 1:
            scores = Y[:, -1]
        else:
            t0 = max(0, Y.shape[1] - int(tail))
            scores = np.nanmean(Y[:, t0:], axis=1)

        v = np.nanmedian(scores)
        if math.isfinite(v) and v < best_val:
            best_val = float(v)
            best_k = k

    return best_k


def plot_mean_ci_comparison(
    results_list: List[Dict[str, Dict[str, Any]]],  # list of exp["results"]
    n: int,
    d_in: int,
    d_out: int,
    steps: int,
    A: torch.Tensor,
    show_grad_norm: bool = True,
    show_grad_cond_num: bool = True,
    title_prefix: str = "",
    savepath: Optional[str] = None,
    show: bool = True,
    # CI controls (kept for call-compatibility; now controls central quantile mass)
    ci_mult: float = 1.96,
    # selection controls
    baseline_patterns: Optional[Dict[str, str]] = None,
    baseline_tail: int = 1,
    muon_family: Optional[str] = None,
    muon_family_pattern: Optional[str] = None,
    # styling
    focus_base_color: str = "tab:blue",
    baseline_color_map: Optional[Dict[str, str]] = None,
    muon_line_alpha: float = 0.90,
    muon_band_alpha: float = 0.25,
    baseline_line_alpha: float = 0.9,
    baseline_band_alpha: float = 0.25,
    # clamp very small values; otherwise plot impossible to read.
    eps: float = 1e-5,
):
    """
    Quantile trajectory plot with bands across experiments (different W0 seeds).

    Center line: median (50th percentile).
    Band: central quantile band determined by ci_mult:
      ci_mult=1.96 -> approx 2.5%..97.5% band.

    Plots:
      - Muon: ONE family (selected) with ALL step sizes (all lrs) as comparisons (blue shades)
      - Baselines: GD best-lr and Adam best-lr (chosen by final median loss)
    """
    if baseline_patterns is None:
        baseline_patterns = {"GD": r"^GD$", "Adam": r"^Adam$"}

    if baseline_color_map is None:
        # default consistent with your earlier preference: GD orange, Adam darker purple
        baseline_color_map = {"GD": "tab:orange", "Adam": "#4B0082"}

    # eigen stats for title
    with torch.no_grad():
        A_sym = 0.5 * (A + A.T)
        evals = torch.linalg.eigvalsh(A_sym).detach().cpu().numpy()
        ev_min = float(evals.min())
        ev_max = float(evals.max())
        kappa = (ev_max / ev_min) if ev_min > 0 else float("inf")

    # quantile choices from ci_mult
    q_lo, q_mid, q_hi = _central_quantiles_from_z(ci_mult)
    band_pct = 100.0 * (q_hi - q_lo)

    # all keys present at least once
    all_keys = sorted({k for res in results_list for k in res.keys()})

    # ---- choose baselines: best-lr by final MEDIAN loss (more consistent with quantiles) ----
    chosen_baselines: Dict[str, str] = {}
    for disp, fam_re in baseline_patterns.items():
        pat = re.compile(fam_re, re.IGNORECASE)
        cands = [k for k in all_keys if pat.search(_family_key(k))]
        best_k = _best_key_by_final_median(results_list, cands, steps, metric="loss", tail=baseline_tail)
        if best_k is not None:
            chosen_baselines[disp] = best_k

    # ---- choose one Muon family and get all its lr variants ----
    fam = _select_muon_family(
        results_list,
        steps=steps,
        muon_family=muon_family,
        muon_family_pattern=muon_family_pattern,
        metric_for_choice="loss",
    )
    if fam is not None:
        fam = fam.strip("_-")

    muon_keys: List[str] = []
    if fam is not None:
        muon_keys = [k for k in all_keys if _family_key(k).strip("_-") == fam]

    # sort muon keys by lr
    muon_keys = sorted(muon_keys, key=lambda k: (_parse_lr(k) is None, _parse_lr(k) or 0.0))
    muon_lrs = [lr for lr in (_parse_lr(k) for k in muon_keys) if lr is not None]

    # ---- axes (loss + optional) ----
    axes_specs = [("loss", "loss (log)", True)]
    if show_grad_norm:
        axes_specs.append(("grad_norm", r"$\|\nabla\|_F$ (log)", True))
    if show_grad_cond_num:
        axes_specs.append(("grad_cond_num", "cond(grad) (log)", True))

    ncols = len(axes_specs)
    fig, axes = plt.subplots(1, ncols, figsize=(10.0 * ncols, 6.2), dpi=140)
    if ncols == 1:
        axes = [axes]
    ax = {name: axes[i] for i, (name, _, _) in enumerate(axes_specs)}

    x = np.arange(steps)

    def _plot_quantile_band(ax_here, mid, lo, hi, color, label,
                            linestyle="-", lw=2.0,
                            line_alpha=0.9, band_alpha=0.18, z=3,
                            log_y=True):
        mid = np.asarray(mid, dtype=float).copy()
        lo  = np.asarray(lo, dtype=float).copy()
        hi  = np.asarray(hi, dtype=float).copy()

        mid[~np.isfinite(mid)] = np.nan
        lo[~np.isfinite(lo)]   = np.nan
        hi[~np.isfinite(hi)]   = np.nan

        # ensure ordering (nan-safe)
        lo = np.minimum(lo, mid)
        hi = np.maximum(hi, mid)

        if log_y:
            mid[mid <= eps] = eps
            lo[lo <= eps]   = eps
            hi[hi <= eps]   = eps

        ax_here.plot(x, mid, color=color, linestyle=linestyle, linewidth=lw,
                     alpha=line_alpha, label=label, zorder=z)
        ax_here.fill_between(x, lo, hi, color=color, alpha=band_alpha,
                             linewidth=0, zorder=z-1)

    # lot Muon (all lrs)
    for k in muon_keys:
        lr = _parse_lr(k)
        col = _shade_for_lr(focus_base_color, lr, muon_lrs)
        lbl = f"{_family_key(k)} | lr={lr:g}" if lr is not None else _family_key(k)

        for metric_name, _, is_log in axes_specs:
            Y = _stack_metric(results_list, k, metric_name, steps)
            if Y.shape[0] == 0:
                continue

            mid, lo, hi = _quantile_band(Y, q_lo=q_lo, q_mid=q_mid, q_hi=q_hi)

            _plot_quantile_band(
                ax[metric_name], mid, lo, hi,
                color=col, label=lbl if metric_name == "loss" else None,
                linestyle="-", lw=2.0,
                line_alpha=muon_line_alpha, band_alpha=muon_band_alpha, z=4,
                log_y=is_log,
            )

    # plot baselines (best lr each)
    for disp, k in chosen_baselines.items():
        lr = _parse_lr(k)
        lr_txt = f"{lr:g}" if lr is not None else "?"
        col = baseline_color_map.get(disp, "k")
        lbl = f"{disp} (best lr={lr_txt})"

        for metric_name, _, is_log in axes_specs:
            Y = _stack_metric(results_list, k, metric_name, steps)
            if Y.shape[0] == 0:
                continue

            mid, lo, hi = _quantile_band(Y, q_lo=q_lo, q_mid=q_mid, q_hi=q_hi)

            _plot_quantile_band(
                ax[metric_name], mid, lo, hi,
                color=col, label=lbl if metric_name == "loss" else None,
                linestyle="--", lw=2.5,
                line_alpha=baseline_line_alpha, band_alpha=baseline_band_alpha, z=10,
                log_y=is_log,
            )

    # formatting
    for metric_name, ylab, is_log in axes_specs:
        a = ax[metric_name]
        if is_log:
            a.set_yscale("log", nonpositive="mask")
        a.set_xlabel("step")
        a.set_ylabel(ylab)
        a.grid(True, which="both", alpha=0.25)


        a.legend(frameon=False, loc="upper right", fontsize=9)
    plt.tight_layout()

    if savepath is not None:
        d = os.path.dirname(savepath)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(savepath, dpi=200, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
