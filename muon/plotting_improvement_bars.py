# muon/plotting_improvement_bars.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from .plotting import _family_key, KIND_RENAME


@dataclass(frozen=True)
class BestRun:
    algo_key: str
    lr: Optional[float]
    init_val: float
    final_val: float


def _to_1d_np(x: Any) -> np.ndarray:
    """Convert list/torch/numpy to 1D float numpy array."""
    if x is None:
        return np.asarray([], dtype=float)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float).reshape(-1)
    return x


def _parse_lr_from_name(name: str) -> Optional[float]:
    """
    Lightweight LR parser (kept local so we don't depend on private helpers).
    Matches substrings like:
      - _lr_0.01
      - _lr0.01
      - -lr=1e-3
    """
    import re

    m = re.search(r"(?:^|[_-])lr[_=]?([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m2 = re.search(r"lr([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", name, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None

    return None


def _score_trace(y: np.ndarray, best_by: str, tail: int) -> Optional[float]:
    """
    Return the scalar score used to pick the best LR run.
    Lower is better.
    """
    if y.size == 0:
        return None

    y = y.astype(float, copy=False)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None

    if best_by == "final":
        return float(y[-1])

    if best_by == "tail_mean":
        t = min(max(int(tail), 1), y.size)
        return float(np.mean(y[-t:]))

    raise ValueError(f"Unknown best_by='{best_by}'")


def select_best_lr_run(
    results: Dict[str, Dict[str, Any]],
    family: str,
    metric: str = "loss",
    best_by: str = "final",
    tail: int = 25,
) -> Optional[BestRun]:
    """
    From a single experiment's `results` dict, select the best LR variant within `family`
    (as defined by muon.plotting._family_key stripping lr tokens).

    Returns BestRun(init, final) from that chosen LR trace, or None if not found.
    """
    best: Optional[BestRun] = None
    best_score: Optional[float] = None

    for algo_key, tr in results.items():
        if _family_key(algo_key) != family:
            continue
        if metric not in tr:
            continue

        y = _to_1d_np(tr.get(metric))
        if y.size == 0:
            continue

        score = _score_trace(y, best_by=best_by, tail=tail)
        if score is None:
            continue

        # pick smallest score (best)
        if best_score is None or score < best_score:
            init_val = float(y[0]) if np.isfinite(y[0]) else float("nan")
            final_val = float(y[-1]) if np.isfinite(y[-1]) else float("nan")
            best = BestRun(
                algo_key=algo_key,
                lr=_parse_lr_from_name(algo_key),
                init_val=init_val,
                final_val=final_val,
            )
            best_score = score

    return best


def median_ranges_best_lr(
    results_list: List[Dict[str, Dict[str, Any]]],
    families: Dict[str, str],
    metric: str = "loss",
    best_by: str = "final",
    tail: int = 25,
) -> Dict[str, Tuple[float, float, int]]:
    """
    Compute (median_init, median_final, n_used) for each display label in `families`.

    IMPORTANT: to keep comparisons fair, we only include an experiment/W0 if it has
    a valid best-lr run for *all* requested families.
    """
    disp_names = list(families.keys())
    init_vals: Dict[str, List[float]] = {d: [] for d in disp_names}
    final_vals: Dict[str, List[float]] = {d: [] for d in disp_names}

    n_used = 0
    for results in results_list:
        picks: Dict[str, BestRun] = {}
        ok = True
        for disp, fam in families.items():
            br = select_best_lr_run(results, family=fam, metric=metric, best_by=best_by, tail=tail)
            if br is None or not np.isfinite(br.init_val) or not np.isfinite(br.final_val):
                ok = False
                break
            picks[disp] = br

        if not ok:
            continue

        # accept this experiment for all families
        n_used += 1
        for disp in disp_names:
            init_vals[disp].append(float(picks[disp].init_val))
            final_vals[disp].append(float(picks[disp].final_val))

    out: Dict[str, Tuple[float, float, int]] = {}
    for disp in disp_names:
        if n_used == 0:
            out[disp] = (float("nan"), float("nan"), 0)
        else:
            out[disp] = (float(np.median(init_vals[disp])), float(np.median(final_vals[disp])), n_used)
    return out


def _kind_entries_sorted_by_gd(
    results_by_kind: Dict[str, List[Dict[str, Dict[str, Any]]]],
    families: Dict[str, str],
    metric: str,
    kind_order: Optional[List[str]],
    kind_rename: Optional[Dict[str, Optional[str]]],
    eps: float,
) -> Tuple[List[dict], List[str]]:
    """
    Build per-kind summaries and sort them by GD performance (worst -> best).

    GD performance measure:
        gd_logratio = log10(max(final,eps)) - log10(max(init,eps))
    (closer to 0 is worse; more negative is better)
    """
    if "GD" not in families:
        raise ValueError("families must include a 'GD' entry to sort kinds by GD performance.")

    if kind_rename is None:
        kind_rename = KIND_RENAME

    disp_names = list(families.keys())

    # choose kind order (input traversal), but final order will be by GD
    all_kinds = list(results_by_kind.keys())
    if kind_order is None:
        traverse = sorted(all_kinds)
    else:
        seen = set(kind_order)
        traverse = [k for k in kind_order if k in results_by_kind]
        traverse += [k for k in all_kinds if k not in seen]

    entries: List[dict] = []
    for kind in traverse:
        rl = results_by_kind.get(kind, [])
        if not rl:
            continue

        label = kind_rename.get(kind, kind)
        if label is None:
            continue  # drop

        stats = median_ranges_best_lr(
            results_list=rl,
            families=families,
            metric=metric,
            best_by="final",
            tail=25,
        )

        # require at least 1 shared experiment for all families
        ok = True
        ini_map: Dict[str, float] = {}
        fin_map: Dict[str, float] = {}
        for disp in disp_names:
            ini, fin, n_used = stats[disp]
            if n_used <= 0 or (not np.isfinite(ini)) or (not np.isfinite(fin)):
                ok = False
                break
            ini_map[disp] = float(ini)
            fin_map[disp] = float(fin)

        if not ok:
            continue

        gd_ini = max(ini_map["GD"], eps)
        gd_fin = max(fin_map["GD"], eps)
        gd_logratio = float(np.log10(gd_fin) - np.log10(gd_ini))
        gd_logratio = min(gd_logratio, 0.0)  # treat increases as "no improvement" (worst)

        entries.append(
            {
                "kind": kind,
                "label": label,
                "ini": ini_map,
                "fin": fin_map,
                "gd_logratio": gd_logratio,
            }
        )

    # worst GD first = largest gd_logratio (closest to 0)
    entries.sort(key=lambda e: e["gd_logratio"], reverse=True)
    return entries, disp_names


def plot_median_improvement_bars(
    results_by_kind,
    families,
    metric: str = "loss",
    kind_order=None,
    kind_rename=None,
    title: str = "Median improvement (best LR per W0)",
    savepath: str | None = None,
    show: bool = False,
    log_y: bool = True,
    linewidth: float = 8.0,
    cap_width: float = 0.10,
    algo_colors=None,
) -> None:
    """
    ALIGNED-TOP plot: bars start at 0 and go downward.
      top    = 0
      bottom = log10(final/init)

    y-axis is integer ticks 0, -1, -2, ... (orders of magnitude decrease).

    Kinds are automatically ordered by GD performance (worst -> best).
    """
    eps = 1e-5  # plotting clamp only

    if algo_colors is None:
        algo_colors = {"GD": "#7A1E3A", "Muon": "#2E5EAA"}

    entries, disp_names = _kind_entries_sorted_by_gd(
        results_by_kind=results_by_kind,
        families=families,
        metric=metric,
        kind_order=kind_order,
        kind_rename=kind_rename,
        eps=eps,
    )
    if len(entries) == 0:
        raise ValueError("No kinds had complete data for all requested families.")

    K = len(entries)
    x0 = np.arange(K, dtype=float) * 1.5
    xlabels = [e["label"] for e in entries]

    m = len(disp_names)
    if m == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.50 / 2, 0.50 / 2, m).tolist()

    fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * K + 2.5), 5.8), dpi=140)

    bottoms_by_disp: Dict[str, np.ndarray] = {}
    tops_by_disp: Dict[str, np.ndarray] = {}

    for disp in disp_names:
        ini = np.array([e["ini"][disp] for e in entries], dtype=float)
        fin = np.array([e["fin"][disp] for e in entries], dtype=float)

        ini_p = np.maximum(ini, eps)
        fin_p = np.maximum(fin, eps)

        if log_y:
            top = np.zeros_like(ini_p)
            bottom = np.log10(fin_p) - np.log10(ini_p)
            bottom = np.minimum(bottom, 0.0)
        else:
            top = np.ones_like(ini_p)
            bottom = fin_p / ini_p
            bottom = np.minimum(bottom, top)

        tops_by_disp[disp] = top
        bottoms_by_disp[disp] = bottom

    for j, disp in enumerate(disp_names):
        xs = x0 + offsets[j]
        y_top = tops_by_disp[disp]
        y_bot = bottoms_by_disp[disp]
        y_bot = np.minimum(y_bot, y_top)

        col = algo_colors.get(disp, None)
        ax.vlines(xs, y_bot, y_top, colors=col, linewidth=linewidth, alpha=0.85, label=disp)
        ax.hlines(y_bot, xs - cap_width, xs + cap_width,
                  colors=col, linewidth=max(1.0, 0.5 * linewidth), alpha=0.95)
        ax.hlines(y_top, xs - cap_width, xs + cap_width,
                  colors=col, linewidth=max(1.0, 0.5 * linewidth), alpha=0.95)

    ax.set_xticks(x0)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.grid(True, which="both", alpha=0.25)

    if log_y:
        ax.set_ylabel("orders of magnitude decrease")
        all_bottoms = np.concatenate([bottoms_by_disp[d] for d in disp_names])
        ymin = float(np.min(all_bottoms))
        ymin_int = int(np.floor(min(ymin, -1e-12)))
        ax.set_yticks(list(range(ymin_int, 1)))  # ..., -2, -1, 0
        ax.set_ylim(ymin_int - 0.25, 0.25)
    else:
        ax.set_ylabel(f"{metric} ratio (final/init)")
        all_bottoms = np.concatenate([bottoms_by_disp[d] for d in disp_names])
        ax.set_ylim(max(0.0, float(np.min(all_bottoms)) - 0.05), 1.05)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=(0, 0, 0.86, 1))

    if savepath is not None:
        d = os.path.dirname(savepath)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(savepath, dpi=200, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_median_absolute_range_bars(
    results_by_kind,
    families,
    metric: str = "loss",
    kind_order=None,
    kind_rename=None,
    title: str = "Median init→final ranges (best LR per W0)",
    savepath: str | None = None,
    show: bool = False,
    log_y: bool = True,
    linewidth: float = 8.0,
    cap_width: float = 0.10,
    algo_colors=None,
) -> None:
    """
    ABSOLUTE-RANGE plot

    For each kind on x-axis, draw one vertical range bar per algorithm family:
        bottom = median(final metric)
        top    = median(init metric)

    Uses plotting-only clamp and (optionally) log y-scale.

    Kinds are automatically ordered by GD performance (worst -> best),
    using the same gd_logratio measure as the aligned-top plot.
    """
    eps = 1e-5  # plotting clamp only

    if algo_colors is None:
        algo_colors = {"GD": "#7A1E3A", "Muon": "#2E5EAA"}

    entries, disp_names = _kind_entries_sorted_by_gd(
        results_by_kind=results_by_kind,
        families=families,
        metric=metric,
        kind_order=kind_order,
        kind_rename=kind_rename,
        eps=eps,
    )
    if len(entries) == 0:
        raise ValueError("No kinds had complete data for all requested families.")

    K = len(entries)
    x0 = np.arange(K, dtype=float) * 1.5
    xlabels = [e["label"] for e in entries]

    m = len(disp_names)
    if m == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.50 / 2, 0.50 / 2, m).tolist()

    fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * K + 2.5), 5.8), dpi=140)

    y_all_lo: List[float] = []
    y_all_hi: List[float] = []

    for j, disp in enumerate(disp_names):
        ini = np.array([e["ini"][disp] for e in entries], dtype=float)
        fin = np.array([e["fin"][disp] for e in entries], dtype=float)

        y_lo = np.maximum(fin, eps)
        y_hi = np.maximum(ini, eps)
        y_hi = np.maximum(y_hi, y_lo)

        y_all_lo.append(float(np.min(y_lo)))
        y_all_hi.append(float(np.max(y_hi)))

        xs = x0 + offsets[j]
        col = algo_colors.get(disp, None)

        ax.vlines(xs, y_lo, y_hi, colors=col, linewidth=linewidth, alpha=0.85, label=disp)
        ax.hlines(y_lo, xs - cap_width, xs + cap_width,
                  colors=col, linewidth=max(1.0, 0.5 * linewidth), alpha=0.95)
        ax.hlines(y_hi, xs - cap_width, xs + cap_width,
                  colors=col, linewidth=max(1.0, 0.5 * linewidth), alpha=0.95)

    ax.set_xticks(x0)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylabel(metric)

    if log_y:
        ax.set_yscale("log", nonpositive="mask")

    ymin = min(y_all_lo)
    ymax = max(y_all_hi)
    if log_y:
        ax.set_ylim(ymin * 0.8, ymax * 1.25)
    else:
        ax.set_ylim(ymin - 0.05 * abs(ymin), ymax + 0.05 * abs(ymax))

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=(0, 0, 0.86, 1))

    if savepath is not None:
        d = os.path.dirname(savepath)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(savepath, dpi=200, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

