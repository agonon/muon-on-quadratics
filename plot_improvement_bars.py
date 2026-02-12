#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import torch

from muon.plotting_improvement_bars import (
    plot_median_improvement_bars,
    plot_median_absolute_range_bars,
)

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


def find_instance_dirs(root: str, match: Optional[str], max_instances: Optional[int]) -> List[str]:
    cands: List[str] = []
    for p in sorted(os.listdir(root)):
        inst_dir = os.path.join(root, p)
        if not os.path.isdir(inst_dir):
            continue
        if match and match not in os.path.basename(inst_dir):
            continue
        if os.path.isfile(os.path.join(inst_dir, "problem.pt")) and os.path.isdir(os.path.join(inst_dir, "experiments")):
            cands.append(inst_dir)

    if max_instances is not None:
        cands = cands[:max_instances]
    return cands


def build_results_by_kind(
    outdir: str,
    match: Optional[str] = None,
    max_instances: Optional[int] = None,
    max_experiments: Optional[int] = None,
    kinds_filter: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    """
    Returns:
        results_by_kind: kind -> list of exp['results'] dicts (one per experiment/W0)

    Notes:
      - Applies KIND_RENAME mapping:
          * if KIND_RENAME[kind] is None -> drop
          * else rename kind to KIND_RENAME[kind]
      - If multiple original kinds map to the same renamed kind, their experiments are merged.
    """
    results_by_kind: Dict[str, List[dict]] = {}

    inst_dirs = find_instance_dirs(outdir, match=match, max_instances=max_instances)
    print(f"[improvement-bars] found {len(inst_dirs)} instance dirs under {outdir}")

    kinds_set = set(kinds_filter) if kinds_filter else None

    for inst_dir in inst_dirs:
        prob_path = os.path.join(inst_dir, "problem.pt")
        exp_dir = os.path.join(inst_dir, "experiments")

        prob = torch.load(prob_path, map_location="cpu")
        kind_raw = prob.get("kind", "unknown")

        if kinds_set is not None and kind_raw not in kinds_set:
            continue

        # apply rename/drop mapping
        kind = KIND_RENAME.get(kind_raw, kind_raw)  # if not in dict, keep original
        if kind is None:
            continue  # explicitly dropped

        print(f'Managed to map {kind_raw} to {kind}')

        exp_files = sorted(glob.glob(os.path.join(exp_dir, "exp_*.pt")))
        if max_experiments is not None:
            exp_files = exp_files[:max_experiments]

        for ep in exp_files:
            exp = torch.load(ep, map_location="cpu")
            if not isinstance(exp, dict) or "results" not in exp:
                continue
            results_by_kind.setdefault(kind, []).append(exp["results"])

    return results_by_kind


def main():
    p = argparse.ArgumentParser(description="Plot median improvement range bars (best LR per W0).")
    p.add_argument("--outdir", type=str, required=True, help="Root directory containing instance dirs.")
    p.add_argument("--match", type=str, default=None, help="Substring filter on instance dir names.")
    p.add_argument("--max_instances", type=int, default=None, help="Limit number of instance dirs scanned.")
    p.add_argument("--max_experiments", type=int, default=None, help="Limit exp_*.pt loaded per instance dir.")
    p.add_argument("--kinds", nargs="*", default=None, help="Optional list of kinds to include (exact match, pre-rename).")
    p.add_argument(
        "--absolute_range",
        action="store_true",
        default=False,
        help="If set, plot absolute init→final ranges (log y). Otherwise, plot aligned-top orders-of-magnitude bars.",
    )


    p.add_argument(
        "--muon_family",
        type=str,
        default="Muon_exact_nest_mom0",
        help="Muon family key (lr-stripped), e.g. Muon_exact_nest_mom0 or Muon_ns5_nest_mom0.",
    )
    p.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Output PDF path. Default: <outdir>/improvement_bars.pdf",
    )
    p.add_argument("--no_show", action="store_true", default=False, help="Do not pop up a window; just save.")
    p.add_argument("--no_logy", action="store_true", default=False, help="Disable log y-scale.")

    args = p.parse_args()

    results_by_kind = build_results_by_kind(
        outdir=args.outdir,
        match=args.match,
        max_instances=args.max_instances,
        max_experiments=args.max_experiments,
        kinds_filter=args.kinds,
    )
    if not results_by_kind:
        raise RuntimeError(
            f"No experiments found under outdir={args.outdir!r} "
            f"(match={args.match!r}, kinds={args.kinds!r})."
        )

    families = {
        "GD": "GD",
        "Muon": args.muon_family,
    }

    savepath = args.savepath or os.path.join(args.outdir, "improvement_bars.pdf")

    plot_fn = plot_median_absolute_range_bars if args.absolute_range else plot_median_improvement_bars

    plot_fn(
        results_by_kind=results_by_kind,
        families=families,
        metric="loss",
        title=f"Median loss improvement (best LR per W0) | Muon={args.muon_family}",
        savepath=savepath,
        show=(not args.no_show),
        log_y=(not args.no_logy),
    )

    

    print(f"[improvement-bars] wrote {savepath}")


if __name__ == "__main__":
    main()
