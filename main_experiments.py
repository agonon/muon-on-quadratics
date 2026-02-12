# main_experiments.py
import os
import re
import math
import json
import time
import glob
import argparse
import random
import time

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import torch

from muon import Muon
from muon.data import build_instance
from muon.runner import compare_traces
from muon.plotting import plot_family_drilldowns
from muon.plotting import plot_traces_side_by_side, plot_mean_ci_comparison


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SchedSpec:
    type: str = "ConstantLR"
    factor: float = 1.0
    total_iters: int = 0

def make_scheduler_from_spec(spec: SchedSpec, steps: int):
    if spec.type == "ConstantLR":
        return lambda opt: torch.optim.lr_scheduler.ConstantLR(
            opt, factor=spec.factor, total_iters=spec.total_iters or steps
        )

    if spec.type == "CosineAnnealingLR":
        return lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=spec.total_iters or steps, eta_min=1e-3
        )

    raise ValueError(f"Unknown scheduler type: {spec.type}")


@dataclass
class AlgoSpec:
    name: str
    opt_type: str  # "SGD" | "Adam" | "Muon"
    lr: float
    momentum: float = 0.0
    nesterov: bool = False
    projection: str = "exact"
    ns_steps: int = 5
    eps: float = 1e-7
    betas: tuple = (0.9, 0.999)

def make_algo_specs(lrs: List[float], include_gd=True, include_muon=True, include_adam=True) -> List[AlgoSpec]:
    specs: List[AlgoSpec] = []
    if include_gd:
        for lr in lrs:
            specs.append(AlgoSpec(name=f"GD_lr_{lr:g}", opt_type="SGD", lr=lr, momentum=0.0))

    if include_muon:
        for lr in lrs:
            for mom in [0, 0.9]:
                for nest in [True]:
                    nest_tag = "nest" if nest else "nonest"
                    for proj in ["exact", "ns"]:
                        if proj == "exact":
                            name = f"Muon_exact_{nest_tag}_mom{mom:g}_lr{lr:g}"
                            specs.append(AlgoSpec(
                                name=name, opt_type="Muon", lr=lr,
                                momentum=mom, nesterov=nest,
                                projection="exact",
                                ns_steps=5
                            ))
                        else:  # "ns"
                            muon_ns_steps = 5
                            name = f"Muon_ns{muon_ns_steps}_{nest_tag}_mom{mom:g}_lr{lr:g}"
                            specs.append(AlgoSpec(
                                name=name, opt_type="Muon", lr=lr,
                                momentum=mom, nesterov=nest,
                                projection="ns",
                                ns_steps=muon_ns_steps
                            ))

    if include_adam:
        for lr in lrs:
            specs.append(AlgoSpec(name=f"Adam_lr_{lr:g}", opt_type="Adam", lr=lr))

    print('All algo specs: ')
    for spec in specs:
        print(spec)

    return specs


def opt_ctor_from_spec(spec: AlgoSpec):
    if spec.opt_type == "SGD":
        return lambda ps: torch.optim.SGD(ps, lr=spec.lr, momentum=spec.momentum)
    if spec.opt_type == "Adam":
        return lambda ps: torch.optim.Adam(ps, lr=spec.lr, betas=spec.betas)
    if spec.opt_type == "Muon":
        return lambda ps: Muon(
            ps,
            lr=spec.lr,
            momentum=spec.momentum,
            nesterov=spec.nesterov,
            projection=spec.projection,
            ns_steps=spec.ns_steps,
            eps=spec.eps,
        )
    raise ValueError(f"Unknown opt_type: {spec.opt_type}")


def _safe(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "_")
    return s

def _fmt_float(x: float) -> str:
    return f"{x:.2e}".replace("+", "").replace(".", "p")

def save_torch(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x

def traces_to_tensors(tr: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(tr)
    for k in ["loss", "grad_norm", "grad_cond_num", "err_F", "err_C"]:
        if k in out and isinstance(out[k], list):
            out[k] = torch.tensor(out[k], dtype=torch.float64)
    return out


def instance_dirname(args, kind: str, instance_seed: int) -> str:
    return _safe(
        f"kind={kind}"
        f"_n={args.n}_din={args.d_in}_dout={args.d_out}"
        f"_smin={_fmt_float(args.s_min)}_smax={_fmt_float(args.s_max)}"
        f"_alpha={args.alpha:g}"
        f"_instseed={instance_seed}"
    )


def run_sweep(args, device: str):
    seed_everything(args.seed + 111)
    W_star = torch.randn(args.d_in, args.d_out, device=device) / math.sqrt(args.d_in)

    lr_scheduler_type = "ConstantLR"
    sched_spec = SchedSpec(type=lr_scheduler_type, factor=1.0, total_iters=args.steps)
    sched_ctor = make_scheduler_from_spec(sched_spec, steps=args.steps)
    print('Created lr scheduler ', lr_scheduler_type)

    algo_specs = make_algo_specs(
        lrs=args.lrs,
        include_gd=(not args.no_gd),
        include_muon=(not args.no_muon),
        include_adam=(not args.no_adam),
    )

    root = args.outdir
    os.makedirs(root, exist_ok=True)

    all_instances_index = []

    for k_idx, kind in enumerate(args.kinds):
        instance_seed = args.seed + 10_000 * (k_idx + 1)

        loss_fn, X, Y, A, kappa_A, A_evals = build_instance(
            kind=kind,
            s_min=args.s_min,
            s_max=args.s_max,
            n=args.n, d_in=args.d_in, d_out=args.d_out,
            device=device,
            W_star=W_star,
            seed=instance_seed,
            alpha=args.alpha,
        )

        inst_dir = os.path.join(root, instance_dirname(args, kind, instance_seed))
        exp_dir = os.path.join(inst_dir, "experiments")
        os.makedirs(exp_dir, exist_ok=True)

        problem = {
            "mode": "sweep",
            "kind": kind,
            "n": args.n,
            "d_in": args.d_in,
            "d_out": args.d_out,
            "s_min": float(args.s_min),
            "s_max": float(args.s_max),
            "alpha": float(args.alpha),
            "global_seed": int(args.seed),
            "instance_seed": int(instance_seed),
            "scale": int(args.n * args.d_out),
            "device_built_on": device,
            "W_star": to_cpu(W_star),
            "X": to_cpu(X),
            "Y": to_cpu(Y),
            "A": to_cpu(A),
            "A_evals": to_cpu(A_evals),
            "kappa_A": float(kappa_A),
            "timestamp": time.time(),
        }
        save_torch(os.path.join(inst_dir, "problem.pt"), problem)

        meta = {
            "instance_dir": inst_dir,
            "problem_file": "problem.pt",
            "scheduler": asdict(sched_spec),
            "steps": args.steps,
            "algo_specs": [asdict(s) for s in algo_specs],
            "num_experiments": args.num_experiments,
            "experiments": [],
        }

        algos = {s.name: opt_ctor_from_spec(s) for s in algo_specs}

        for exp_id in range(args.num_experiments):
            start = time.time()

            init_seed = args.seed + 1_000_000 + 10_000 * (k_idx + 1) + exp_id
            seed_everything(init_seed)
            W0 = torch.randn(args.d_in, args.d_out, device=device) / math.sqrt(args.d_in)

            results = compare_traces(loss_fn, [W0], algos, args.steps, sched_ctor=sched_ctor)

            results_cpu = {}
            for name, tr in results.items():
                tr2 = traces_to_tensors(tr)
                for kk, vv in list(tr2.items()):
                    tr2[kk] = to_cpu(vv)
                results_cpu[name] = tr2

            exp_record = {
                "exp_id": exp_id,
                "init_seed": int(init_seed),
                "W0": to_cpu(W0),
                "results": results_cpu,
                "timestamp": time.time(),
            }

            exp_file = f"exp_{exp_id:04d}.pt"
            save_torch(os.path.join(exp_dir, exp_file), exp_record)

            finals = {name: float(tr["loss"][-1].item()) for name, tr in results_cpu.items()}
            best = min(finals, key=finals.get)
            print(
                f"[SWEEP kind={kind:12s} exp={exp_id:02d}/{args.num_experiments-1:02d} "
                f"kappa(A)≈{kappa_A:.1e}] best={best} ({finals[best]:.3e})"
            )

            meta["experiments"].append({
                "exp_id": exp_id,
                "exp_file": os.path.join("experiments", exp_file),
                "init_seed": int(init_seed),
                "best_algo": best,
                "best_final_loss": float(finals[best]),
            })

            end = time.time()
            print('Total time per experiment: ', end - start)

        save_json(os.path.join(inst_dir, "meta.json"), meta)
        all_instances_index.append({
            "kind": kind,
            "instance_seed": int(instance_seed),
            "instance_dir": inst_dir,
            "kappa_A": float(kappa_A),
        })

    save_json(os.path.join(root, "instances_index.json"), all_instances_index)


def run_plot(args):
    # headless backend if no_show (important for batch runs / servers)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    from muon.plotting import plot_traces_side_by_side

    root = args.outdir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--outdir '{root}' not found")

    # find instance dirs = those that contain problem.pt and experiments/
    candidates = []
    for p in sorted(os.listdir(root)):
        inst_dir = os.path.join(root, p)
        if not os.path.isdir(inst_dir):
            continue
        if os.path.isfile(os.path.join(inst_dir, "problem.pt")) and os.path.isdir(os.path.join(inst_dir, "experiments")):
            candidates.append(inst_dir)

    # filtering by substring
    if args.match:
        candidates = [d for d in candidates if args.match in os.path.basename(d)]

    if args.max_instances is not None:
        candidates = candidates[:args.max_instances]

    print(f"[PLOT] found {len(candidates)} instance dirs under {root}")

    for inst_dir in candidates:
        problem = torch.load(os.path.join(inst_dir, "problem.pt"))
        A = problem["A"]
        n = int(problem["n"])
        d_in = int(problem["d_in"])
        d_out = int(problem["d_out"])

        meta_path = os.path.join(inst_dir, "meta.json")
        steps = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            steps = int(meta.get("steps", 0)) or None

        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if args.max_experiments is not None:
            exp_files = exp_files[:args.max_experiments]

        if args.plot_outdir:
            base = os.path.join(args.plot_outdir, os.path.basename(inst_dir))
        else:
            base = inst_dir
        plot_dir = os.path.join(base, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        print(f"[PLOT] instance={os.path.basename(inst_dir)} exps={len(exp_files)} -> {plot_dir}")

        
        # mean ± CI plot (across experiments)
        results_list = []
        for exp_path in exp_files:
            exp = torch.load(exp_path, map_location="cpu")
            results_list.append(exp["results"])

        # infer steps if needed (use first available)
        if steps is None and results_list:
            first = next(iter(results_list[0].values()))
            steps = int(len(first["loss"]))

        kind = problem.get("kind", "unknown")
        mean_savepath = os.path.join(plot_dir, f"{kind}_mean_ci.pdf")
        mean_title = f"kind={kind} | {os.path.basename(inst_dir)} | mean±CI"

        plot_mean_ci_comparison(
            results_list,
            n=n, d_in=d_in, d_out=d_out,
            steps=steps,
            A=A,
            show_grad_norm=(not args.hide_grad_norm),
            show_grad_cond_num=(not args.hide_grad_cond),
            title_prefix=mean_title,
            savepath=mean_savepath,
            show=(not args.no_show),
            ci_mult=1.96,
            # pick best lr baselines by last-iterate mean loss:
            baseline_patterns={"GD": r"^GD$", "Adam": r"^Adam$"},
            baseline_tail=1,
            # choose ONE Muon family:
            # set muon_family_pattern to force one, e.g. r"^Muon_ns5_nest_mom0\.9$"
            # or leave both None to auto-pick best Muon family by final mean loss
            muon_family_pattern=None,
            muon_family="Muon_exact_nest_mom0",
            focus_base_color="tab:blue",
            baseline_color_map={"GD": "tab:orange", "Adam": "#4B0082"},
            muon_line_alpha=0.90,
            muon_band_alpha=0.18,
            baseline_line_alpha=0.95,
            baseline_band_alpha=0.18,
        )

        # Separate plots for each experiment.
        if args.plot_separate:
            print("PLOTTING SEPARATELY")
            for exp_path in exp_files:
                exp = torch.load(exp_path)
                exp_id = int(exp.get("exp_id", -1))
                results = exp["results"]

                # infer steps if needed
                if steps is None:
                    first = next(iter(results.values()))
                    steps = int(len(first["loss"]))

                savepath = os.path.join(plot_dir, f"kind={kind}_exp_{exp_id:04d}.pdf")
                kind = problem.get("kind", "unknown")
                title_prefix = f"kind={kind} | {os.path.basename(inst_dir)} | exp={exp_id:04d}"

                plot_family_drilldowns(
                    results,
                    n=n, d_in=d_in, d_out=d_out,
                    steps=steps,
                    A=A,
                    title_prefix=title_prefix,
                    out_dir=os.path.join(plot_dir, f"exp_{exp_id:04d}"),
                    show=(not args.no_show),
                    show_grad_norm=(not args.hide_grad_norm),
                    show_grad_cond_num=(not args.hide_grad_cond),
                    focus_base_color="tab:blue",
                    baseline_color_map={"GD": "tab:orange", "Adam": "#3B0A45"},
                )



def _tensor_info(name: str, t):
    if not torch.is_tensor(t):
        print(f"  {name}: {t}")
        return
    t = t.detach()
    mn = float(t.min().item()) if t.numel() else float("nan")
    mx = float(t.max().item()) if t.numel() else float("nan")
    nrm = float(t.norm().item()) if t.numel() else float("nan")
    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
          f"min={mn:.3e} max={mx:.3e} norm={nrm:.3e}")


def _find_instance_dirs(root: str, match: Optional[str] = None):
    cands = []
    for p in sorted(os.listdir(root)):
        inst_dir = os.path.join(root, p)
        if not os.path.isdir(inst_dir):
            continue
        if match and (match not in os.path.basename(inst_dir)):
            continue
        if os.path.isfile(os.path.join(inst_dir, "problem.pt")) and os.path.isdir(os.path.join(inst_dir, "experiments")):
            cands.append(inst_dir)
    return cands


def _mean_and_se(xs: List[float]) -> Tuple[float, float, int]:
    """
    mean and standard error:
      SE = s / sqrt(n),  s = sample std with ddof=1 (unbiased), n=len(xs)
    """
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    x = torch.tensor(xs, dtype=torch.float64)
    mean = float(x.mean().item())
    if n >= 2:
        std = float(x.std(unbiased=True).item())  # ddof=1
        se = std / math.sqrt(n)
    else:
        se = 0.0
    return mean, se, n


def _trace_get_at(v, idx: int) -> float:
    if torch.is_tensor(v):
        return float(v[idx].item())
    return float(v[idx])


def _infer_available_algos(root: str, kinds: List[str], match: Optional[str], max_instances: Optional[int]) -> List[str]:
    inst_dirs = _find_instance_dirs(root, match=match)
    if max_instances is not None:
        inst_dirs = inst_dirs[:max_instances]
    for inst_dir in inst_dirs:
        prob = torch.load(os.path.join(inst_dir, "problem.pt"), map_location="cpu")
        kind = prob.get("kind", None)
        if kind is not None and kind not in kinds:
            continue
        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if not exp_files:
            continue
        exp = torch.load(exp_files[0], map_location="cpu")
        return sorted(list(exp.get("results", {}).keys()))
    return []


def _resolve_algo_keys(requested: Optional[List[str]], available: List[str]) -> List[str]:
    """
    requested items are matched as:
      - if starts with 're:' => regex
      - else: exact match if exists, otherwise substring match
    """
    if not requested:
        return available

    out: List[str] = []
    for spec in requested:
        matches: List[str] = []
        if spec.startswith("re:"):
            pat = re.compile(spec[3:])
            matches = [k for k in available if pat.search(k)]
        else:
            if spec in available:
                matches = [spec]
            else:
                matches = [k for k in available if spec in k]

        if not matches:
            print(f"[TABLE] warning: no algo matches '{spec}'")
        out.extend(matches)

    # unique preserve order
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def _match_keys(pattern: str, available: List[str]) -> List[str]:
    """
    pattern:
      - "re:<regex>" => regex search over keys
      - exact match if present
      - otherwise substring match
    """
    if pattern.startswith("re:"):
        pat = re.compile(pattern[3:])
        return [k for k in available if pat.search(k)]
    if pattern in available:
        return [pattern]
    return [k for k in available if pattern in k]


def _loss_at(loss_trace, idx: int) -> float:
    return _trace_get_at(loss_trace, idx)


def run_table(args):
    root = args.outdir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--outdir '{root}' not found")

    inst_dirs = _find_instance_dirs(root, match=args.match)
    if args.max_instances is not None:
        inst_dirs = inst_dirs[:args.max_instances]
    if not inst_dirs:
        print("[TABLE] no instance dirs found")
        return

    # find available keys from the first experiment we can load
    available_keys = None
    for inst_dir in inst_dirs:
        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if not exp_files:
            continue
        exp0 = torch.load(exp_files[0], map_location="cpu")
        available_keys = sorted(list(exp0.get("results", {}).keys()))
        break
    if not available_keys:
        print("[TABLE] could not infer available keys (no experiments?)")
        return

    # Parse groups: NAME=PATTERN
    if not args.table_groups:
        print("[TABLE] ERROR: please pass --table_groups (NAME=PATTERN ...).")
        print("Example: --table_groups GD=GD_lr_ MuonExactNoMom=Muon_exact_nest_mom0_")
        return

    groups: List[Tuple[str, str, List[str]]] = []
    for g in args.table_groups:
        if "=" not in g:
            raise ValueError(f"--table_groups entry must be NAME=PATTERN, got '{g}'")
        name, pat = g.split("=", 1)
        cands = _match_keys(pat, available_keys)
        if not cands:
            print(f"[TABLE] warning: group '{name}' pattern '{pat}' matched nothing")
        else:
            print('Candidates: ', cands)
        groups.append((name, pat, cands))

    print("[TABLE] groups:")
    for name, pat, cands in groups:
        print(f"  - {name}: pattern='{pat}' matches {len(cands)} keys")

    T = int(args.steps)
    idx_t10 = max(0, T // 10)
    idx_t2  = max(0, T // 2)
    ci = float(args.table_ci)

    # vals[kind][group]["t10"/"t2"/"tend"] = list over experiments of best-lr chosen values
    vals: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    chosen_key_counts: Dict[str, Dict[str, Dict[str, int]]] = {}  # kind->group->key->count

    for inst_dir in inst_dirs:
        prob = torch.load(os.path.join(inst_dir, "problem.pt"), map_location="cpu")
        kind = prob.get("kind", "unknown")
        if kind not in args.kinds:
            continue

        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if args.max_experiments is not None:
            exp_files = exp_files[:args.max_experiments]

        for ef in exp_files:
            exp = torch.load(ef, map_location="cpu")
            results = exp.get("results", {})

            for group_name, pat, cands in groups:
                # candidates that actually exist in this experiment file
                present = []
                for k in cands:
                    tr = results.get(k, None)
                    if tr is None:
                        continue
                    if "loss" not in tr:
                        continue
                    present.append(k)

                if not present:
                    continue

                # pick best lr = best final loss
                def final_loss(key: str) -> float:
                    loss_trace = results[key]["loss"]
                    L = int(loss_trace.numel()) if torch.is_tensor(loss_trace) else len(loss_trace)
                    return _loss_at(loss_trace, L - 1)

                best_key = min(present, key=final_loss)
                loss_trace = results[best_key]["loss"]
                L = int(loss_trace.numel()) if torch.is_tensor(loss_trace) else len(loss_trace)
                idx_end = max(0, L - 1)

                v10  = _loss_at(loss_trace, min(idx_t10, idx_end))
                v2   = _loss_at(loss_trace, min(idx_t2,  idx_end))
                vend = _loss_at(loss_trace, idx_end)

                vals.setdefault(kind, {}).setdefault(group_name, {}).setdefault("t10", []).append(v10)
                vals.setdefault(kind, {}).setdefault(group_name, {}).setdefault("t2", []).append(v2)
                vals.setdefault(kind, {}).setdefault(group_name, {}).setdefault("tend", []).append(vend)

                chosen_key_counts.setdefault(kind, {}).setdefault(group_name, {}).setdefault(best_key, 0)
                chosen_key_counts[kind][group_name][best_key] += 1

    # Print one table per group (rows=kinds)
    print("\n" + "=" * 90)
    print(f"[TABLE] best-lr-per-W0. loss shown as mean ± {ci}·SE")
    print(f"[TABLE] indices: t10={idx_t10}, t2={idx_t2}, tend=last")
    print("=" * 90)

    for group_name, pat, _ in groups:
        print("\n" + "-" * 90)
        print(f"GROUP: {group_name}   (pattern='{pat}')")
        print(f"{'kind':18s} | {'t=T/10':22s} | {'t=T/2':22s} | {'t=T':22s}")
        print("-" * 90)

        for kind in args.kinds:
            d = vals.get(kind, {}).get(group_name, {})
            m10, se10, n10 = _mean_and_se(d.get("t10", []))
            m2,  se2,  n2  = _mean_and_se(d.get("t2", []))
            me,  see,  ne  = _mean_and_se(d.get("tend", []))

            def fmt(m, se, n):
                if not math.isfinite(m):
                    return "NA"
                return f"{m:.3e} ± {(ci*se):.2e} (n={n})"

            print(f"{kind:18s} | {fmt(m10,se10,n10):22s} | {fmt(m2,se2,n2):22s} | {fmt(me,see,ne):22s}")

        # optional: show which lrs got picked most often
        if args.table_out_json is None:
            # keep stdout short; only show top 3 picks per kind
            for kind in args.kinds:
                counts = chosen_key_counts.get(kind, {}).get(group_name, {})
                if not counts:
                    continue
                top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top_str = ", ".join([f"{k}×{c}" for k, c in top])
                print(f"  [picks] {kind}: {top_str}")

    if args.table_out_json:
        out = {
            "steps_arg": int(args.steps),
            "idx_t10": int(idx_t10),
            "idx_t2": int(idx_t2),
            "ci_mult": float(ci),
            "kinds": list(args.kinds),
            "groups": [{"name": g[0], "pattern": g[1]} for g in groups],
            "vals": vals,  # raw lists
            "chosen_key_counts": chosen_key_counts,
        }
        save_json(args.table_out_json, out)
        print(f"\n[TABLE] wrote: {args.table_out_json}")


def _best_so_far_value(loss_trace, idx: int) -> float:
    """
    Return min_{s<=idx} loss_trace[s], treating non-finite as +inf.
    """
    if not torch.is_tensor(loss_trace):
        loss_trace = torch.tensor(loss_trace, dtype=torch.float64)
    else:
        loss_trace = loss_trace.to(dtype=torch.float64)

    if loss_trace.numel() == 0:
        return float("nan")

    # clamp idx
    idx = int(max(0, min(idx, loss_trace.numel() - 1)))

    # replace NaN/inf with +inf so they never get selected as best
    inf = torch.tensor(float("inf"), dtype=loss_trace.dtype)
    cleaned = torch.where(torch.isfinite(loss_trace), loss_trace, inf)

    best_prefix = torch.cummin(cleaned, dim=0)[0]
    v = float(best_prefix[idx].item())
    return v


def _best_group_value_by_time(results: Dict[str, Any], keys: List[str], idx: int) -> Tuple[float, Optional[str]]:
    """
    For a group (list of algo keys), compute:
        min_{key in keys}  min_{s<=idx} loss_key[s]
    Returns (best_value, best_key).
    """
    best_v = float("inf")
    best_k = None
    for k in keys:
        tr = results.get(k, None)
        if tr is None or "loss" not in tr:
            continue
        v = _best_so_far_value(tr["loss"], idx)
        if math.isfinite(v) and (v < best_v):
            best_v = v
            best_k = k

    if best_k is None:
        return float("nan"), None
    return best_v, best_k


def _winrate_ci(wins: int, n: int, ci_mult: float = 1.96) -> Tuple[float, float]:
    """
    Wald CI around win-rate p = wins/n:
      SE = sqrt(p(1-p)/n)
      CI half-width = ci_mult * SE
    Returns (p, halfwidth).
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = wins / n
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")
    return p, ci_mult * se


def run_table_ratios_same_dir(args):
    """
    Compare TWO optimizer families (defined by --table_groups) by:
      - win-rate (A better than B) using log-ratios for numerical stability
      - mean log-ratio (log A - log B) with standard error

    For each kind, each init (=each exp file), and each time index t:
      A_t = min_{LR in groupA} min_{s<=t} loss(LR)[s]
      B_t = min_{LR in groupB} min_{s<=t} loss(LR)[s]

      log_ratio = log(A_t + eps) - log(B_t + eps)

      A wins if log_ratio < -tol
      B wins if log_ratio > +tol
      tie otherwise

    Win-rate CI: Wald ± z*sqrt(p(1-p)/n) with z=args.table_ci (default 1.96).
    """

    root = args.outdir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--outdir '{root}' not found")

    inst_dirs = _find_instance_dirs(root, match=args.match)
    if args.max_instances is not None:
        inst_dirs = inst_dirs[:args.max_instances]
    if not inst_dirs:
        print("[TABLE-RATIOS] no instance dirs found")
        return

    # Find available keys from first experiment we can load
    available_keys = None
    for inst_dir in inst_dirs:
        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if not exp_files:
            continue
        exp0 = torch.load(exp_files[0], map_location="cpu")
        available_keys = sorted(list(exp0.get("results", {}).keys()))
        break
    if not available_keys:
        print("[TABLE-RATIOS] could not infer available keys (no experiments?)")
        return

    if not args.table_groups or len(args.table_groups) != 2:
        print("[TABLE-RATIOS] ERROR: please pass exactly TWO groups via --table_groups.")
        print("Example: --table_groups GD=GD_lr_ Muon=Muon_exact_")
        return

    # Parse two groups: NAME=PATTERN
    parsed = []
    for g in args.table_groups:
        if "=" not in g:
            raise ValueError(f"--table_groups entry must be NAME=PATTERN, got '{g}'")
        name, pat = g.split("=", 1)
        keys = _match_keys(pat, available_keys)
        if not keys:
            print(f"[TABLE-RATIOS] warning: group '{name}' pattern '{pat}' matched nothing")
        parsed.append((name, pat, keys))

    (nameA, patA, keysA), (nameB, patB, keysB) = parsed

    print("[TABLE-RATIOS] comparing:")
    print(f"  A = {nameA} (pattern='{patA}') -> {len(keysA)} keys")
    print(f"  B = {nameB} (pattern='{patB}') -> {len(keysB)} keys")

    z = float(args.table_ci)              # e.g. 1.96
    eps = float(getattr(args, "ratio_eps", 1e-300))
    tol = float(getattr(args, "ratio_tol", 0.0))


    # Evaluation times (best loss in [0..t])
    T_arg = int(args.steps)
    idx_targets = {
        "t10":  max(0, T_arg // 10),
        "t2":   max(0, T_arg // 2),
        "tend": max(0, T_arg - 1),
    }

    # stats[kind][label] = dict(winsA, winsB, ties, n, log_ratios)
    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _wald_ci_halfwidth(p_hat: float, n: int) -> float:
        if n <= 0 or not math.isfinite(p_hat):
            return float("nan")
        se = math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / n)
        return z * se

    for inst_dir in inst_dirs:
        prob = torch.load(os.path.join(inst_dir, "problem.pt"), map_location="cpu")
        kind = prob.get("kind", "unknown")
        if kind not in args.kinds:
            continue

        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if args.max_experiments is not None:
            exp_files = exp_files[:args.max_experiments]

        for ef in exp_files:
            exp = torch.load(ef, map_location="cpu")
            results = exp.get("results", {})
            if not results:
                continue

            for label, idx in idx_targets.items():
                a, _ = _best_group_value_by_time(results, keysA, idx)
                b, _ = _best_group_value_by_time(results, keysB, idx)

                # must be finite and nonnegative to be meaningful for log
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                a = max(a, 0)
                b = max(b, 0)

                lr = math.log(a + eps) - math.log(b + eps)
                ratio = (a + eps) / (b + eps)

                rec = stats.setdefault(kind, {}).setdefault(
                    label,
                    {"winsA": 0, "winsB": 0, "ties": 0, "n": 0, "log_ratios": [], "ratios": []},
                )
                rec["n"] += 1

                if lr < -tol:
                    rec["winsA"] += 1
                elif lr > tol:
                    rec["winsB"] += 1
                else:
                    rec["ties"] += 1

                if math.isfinite(lr):
                    rec["log_ratios"].append(float(lr))
                    rec["ratios"].append(float(ratio))


    # Print summary
    print("\n" + "=" * 120)
    print(f"[TABLE-RATIOS] A vs B win-rate using best-by-time across LR (best loss in [0..t])")
    print(f"[TABLE-RATIOS] A={nameA}  vs  B={nameB}")
    print(f"[TABLE-RATIOS] log_ratio = log(A+eps) - log(B+eps), eps={eps:.1e}, tol={tol:g}")
    print("=" * 120)

    colw = 28
    print(f"{'kind':30s} | {'t=T/10':{colw}s} | {'t=T/2':{colw}s} | {'t=T':{colw}s}")
    print("-" * 120)

    # Use this one for win stats.
    def fmt_cell(rec: Optional[Dict[str, Any]]) -> str:
        if not rec or rec["n"] <= 0:
            return "NA"
        n = int(rec["n"])
        winsA = int(rec["winsA"])
        winsB = int(rec["winsB"])
        ties = int(rec["ties"])

        p_hat = winsA / n
        hw = _wald_ci_halfwidth(p_hat, n)

        # mean log-ratio and its SE 
        m_lr, se_lr, n_lr = _mean_and_se(rec.get("log_ratios", []))
        lr_hw = z * se_lr if math.isfinite(se_lr) else float("nan")

        s = f"winA={100*p_hat:5.1f}%"
        
        return s

    for kind in args.kinds:
        r10 = stats.get(kind, {}).get("t10", None)
        r2  = stats.get(kind, {}).get("t2", None)
        re  = stats.get(kind, {}).get("tend", None)
        print(f"{kind:30s} | {fmt_cell(r10):{colw}s} | {fmt_cell(r2):{colw}s} | {fmt_cell(re):{colw}s}")

    if args.table_out_json:
        out = {
            "mode": "table-ratios",
            "A": {"name": nameA, "pattern": patA, "keys": keysA},
            "B": {"name": nameB, "pattern": patB, "keys": keysB},
            "idx_targets": idx_targets,
            "z": float(z),
            "eps": float(eps),
            "tol": float(tol),
            "kinds": list(args.kinds),
            "stats": stats,
        }
        save_json(args.table_out_json, out)
        print(f"\n[TABLE-RATIOS] wrote: {args.table_out_json}")


def run_table_ratios(args):
    """
    Compare TWO optimizer families (defined by --table_groups) by:
      - win-rate (A better than B) using log-ratios for numerical stability
      - (optionally stored) log-ratios and ratios

      If --outdir_b is provided, then:
        - group A is evaluated on args.outdir
        - group B is evaluated on args.outdir_b
      Instances are paired by instance-dir basename.
      Experiments are paired by exp_XXXX.pt basename.

      If --outdir_b is None, both groups are read from args.outdir.
    """

    rootA = args.outdir
    rootB = getattr(args, "outdir_b", None)

    # original single root path.
    if not rootB:
        print('Producing table from one directory.')
        run_table_ratios_same_dir(args)
        return
    
    # cross root path.
    if not os.path.isdir(rootA):
        raise FileNotFoundError(f"--outdir '{rootA}' not found")
    if not os.path.isdir(rootB):
        raise FileNotFoundError(f"--outdir_b '{rootB}' not found")

    inst_dirsA = _find_instance_dirs(rootA, match=args.match)
    inst_dirsB = _find_instance_dirs(rootB, match=args.match)

    if args.max_instances is not None:
        inst_dirsA = inst_dirsA[:args.max_instances]
        inst_dirsB = inst_dirsB[:args.max_instances]

    if not inst_dirsA or not inst_dirsB:
        print("[TABLE-RATIOS] no instance dirs found in one (or both) outdirs")
        return

    def _pair_key(inst_dir: str) -> str:
        import re

        base = os.path.basename(inst_dir)

        # if there are suffixes like "..._twoeig"
        base = re.sub(r"_twoeig$", "", base)

        # drop the instseed, since it can differ across outdirs
        base = re.sub(r"_instseed=\d+", "", base)

        # normalize any accidental double-underscores
        base = re.sub(r"__+", "_", base).strip("_-")
        return base

    # build multi-map for B (in case of duplicates)
    mapB: Dict[str, List[str]] = {}
    for d in inst_dirsB:
        k = _pair_key(d)
        mapB.setdefault(k, []).append(d)

    # pair A->B by key; if duplicates exist in B, pop one deterministically
    paired: List[Tuple[str, str]] = []
    unpairedA: List[str] = []

    for dA in inst_dirsA:
        k = _pair_key(dA)
        lst = mapB.get(k, [])
        if lst:
            dB = lst.pop(0)
            paired.append((dA, dB))
        else:
            unpairedA.append(dA)

    # diagnostics
    if unpairedA:
        print("[TABLE-RATIOS] Unpaired instances from outdir A (after stripping instseed):")
        for d in unpairedA[:10]:
            print("  -", os.path.basename(d), "-> key:", _pair_key(d))
        if len(unpairedA) > 10:
            print(f"  ... ({len(unpairedA)-10} more)")

    # also show any leftover B instances that never got used
    leftoverB = [(k, v) for k, v in mapB.items() if v]
    if leftoverB:
        print("[TABLE-RATIOS] Extra instances in outdir B not paired:")
        for k, v in leftoverB[:10]:
            print("  - key:", k, "examples:", [os.path.basename(x) for x in v[:2]])
        if len(leftoverB) > 10:
            print(f"  ... ({len(leftoverB)-10} more)")

    if not paired:
        print("[TABLE-RATIOS] found 0 paired instances by directory basename.")
        print("  If your two outdirs were generated with different instance_dirname strings,")
        print("  you’ll need a more robust pairing key (e.g. match on problem.pt fields).")
        return

    # infer available keys separately for A and B
    def _infer_available_keys_from_root(inst_dir: str) -> List[str]:
        exp_files = sorted(glob.glob(os.path.join(inst_dir, "experiments", "exp_*.pt")))
        if not exp_files:
            return []
        exp0 = torch.load(exp_files[0], map_location="cpu")
        return sorted(list(exp0.get("results", {}).keys()))

    available_keysA = _infer_available_keys_from_root(paired[0][0])
    available_keysB = _infer_available_keys_from_root(paired[0][1])

    if not available_keysA:
        print("[TABLE-RATIOS] could not infer available keys from outdir A (no experiments?)")
        return
    if not available_keysB:
        print("[TABLE-RATIOS] could not infer available keys from outdir B (no experiments?)")
        return

    if not args.table_groups or len(args.table_groups) != 2:
        print("[TABLE-RATIOS] ERROR: please pass exactly TWO groups via --table_groups.")
        print("Example: --table_groups Muon=Muon_exact_ GD=GD_lr_")
        return

    # Parse two groups: NAME=PATTERN
    parsed = []
    for g in args.table_groups:
        if "=" not in g:
            raise ValueError(f"--table_groups entry must be NAME=PATTERN, got '{g}'")
        name, pat = g.split("=", 1)
        parsed.append((name, pat))

    (nameA, patA), (nameB, patB) = parsed
    keysA = _match_keys(patA, available_keysA)
    keysB = _match_keys(patB, available_keysB)

    if not keysA:
        print(f"[TABLE-RATIOS] warning: group A '{nameA}' pattern '{patA}' matched nothing in outdir A")
    if not keysB:
        print(f"[TABLE-RATIOS] warning: group B '{nameB}' pattern '{patB}' matched nothing in outdir B")

    print("[TABLE-RATIOS] comparing across TWO outdirs:")
    print(f"  outdir A = {rootA}")
    print(f"  outdir B = {rootB}")
    print(f"  A = {nameA} (pattern='{patA}') -> {len(keysA)} keys in A")
    print(f"  B = {nameB} (pattern='{patB}') -> {len(keysB)} keys in B")
    print(f"  paired instances: {len(paired)}")

    z = float(args.table_ci)
    eps = float(getattr(args, "ratio_eps", 1e-300))
    tol = float(getattr(args, "ratio_tol", 0.0))

    T_arg = int(args.steps)
    idx_targets = {
        "t10":  max(0, T_arg // 10),
        "t2":   max(0, T_arg // 2),
        "tend": max(0, T_arg - 1),
    }

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _wald_ci_halfwidth(p_hat: float, n: int) -> float:
        if n <= 0 or not math.isfinite(p_hat):
            return float("nan")
        se = math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / n)
        return z * se

    # Iterate paired instances
    for instA, instB in paired:
        probA = torch.load(os.path.join(instA, "problem.pt"), map_location="cpu")
        kind = probA.get("kind", "unknown")
        if kind not in args.kinds:
            continue

        exp_filesA = sorted(glob.glob(os.path.join(instA, "experiments", "exp_*.pt")))
        exp_filesB = sorted(glob.glob(os.path.join(instB, "experiments", "exp_*.pt")))

        if not exp_filesA or not exp_filesB:
            continue

        # Pair experiments by filename basename (exp_0007.pt etc.)
        map_expB = {os.path.basename(p): p for p in exp_filesB}
        common = [pA for pA in exp_filesA if os.path.basename(pA) in map_expB]
        common = sorted(common, key=lambda p: os.path.basename(p))

        if args.max_experiments is not None:
            common = common[:args.max_experiments]

        for pA in common:
            pB = map_expB[os.path.basename(pA)]

            expA = torch.load(pA, map_location="cpu")
            expB = torch.load(pB, map_location="cpu")

            resultsA = expA.get("results", {})
            resultsB = expB.get("results", {})
            if not resultsA or not resultsB:
                continue

            for label, idx in idx_targets.items():
                a, _ = _best_group_value_by_time(resultsA, keysA, idx)
                b, _ = _best_group_value_by_time(resultsB, keysB, idx)

                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                a = max(a, 0)
                b = max(b, 0)

                lr = math.log(a + eps) - math.log(b + eps)
                ratio = (a + eps) / (b + eps)

                rec = stats.setdefault(kind, {}).setdefault(
                    label,
                    {"winsA": 0, "winsB": 0, "ties": 0, "n": 0, "log_ratios": [], "ratios": []},
                )
                rec["n"] += 1

                if lr < -tol:
                    rec["winsA"] += 1
                elif lr > tol:
                    rec["winsB"] += 1
                else:
                    rec["ties"] += 1

                if math.isfinite(lr):
                    rec["log_ratios"].append(float(lr))
                    rec["ratios"].append(float(ratio))
                else:
                    print(f"log ratio is not finite, a = {a}, b = {b}")

    # Print summary
    print("\n" + "=" * 120)
    print(f"[TABLE-RATIOS] A vs B win-rate using best-by-time across LR (best loss in [0..t])")
    print(f"[TABLE-RATIOS] outdir A={rootA}")
    print(f"[TABLE-RATIOS] outdir B={rootB}")
    print(f"[TABLE-RATIOS] A={nameA}  vs  B={nameB}")
    # print(f"[TABLE-RATIOS] log_ratio = log(A+eps) - log(B+eps), eps={eps:.1e}, tol={tol:g}")
    print("=" * 120)

    colw = 28
    print(f"{'kind':30s} | {'t=T/10':{colw}s} | {'t=T/2':{colw}s} | {'t=T':{colw}s}")
    print("-" * 120)

    def fmt_cell(rec: Optional[Dict[str, Any]]) -> str:
        if not rec or rec["n"] <= 0:
            return "NA"
        n = int(rec["n"])
        winsA = int(rec["winsA"])
        p_hat = winsA / n
        hw = _wald_ci_halfwidth(p_hat, n)
        return f"winA={100*p_hat:5.1f}%"

    for kind in args.kinds:
        r10 = stats.get(kind, {}).get("t10", None)
        r2  = stats.get(kind, {}).get("t2", None)
        re  = stats.get(kind, {}).get("tend", None)
        print(f"{kind:30s} | {fmt_cell(r10):{colw}s} | {fmt_cell(r2):{colw}s} | {fmt_cell(re):{colw}s}")

    if args.table_out_json:
        out = {
            "mode": "table-ratios",
            "outdirA": rootA,
            "outdirB": rootB,
            "A": {"name": nameA, "pattern": patA, "keys": keysA},
            "B": {"name": nameB, "pattern": patB, "keys": keysB},
            "idx_targets": idx_targets,
            "z": float(z),
            "eps": float(eps),
            "tol": float(tol),
            "kinds": list(args.kinds),
            "stats": stats,
        }
        save_json(args.table_out_json, out)
        print(f"\n[TABLE-RATIOS] wrote: {args.table_out_json}")


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--mode", choices=["sweep", "plot", "table", "table-ratios"], default="sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--d_in", type=int, default=128)
    p.add_argument("--d_out", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)

    p.add_argument("--s_min", type=float, default=1e-3)
    p.add_argument("--s_max", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument(
        "--outdir_b", type=str, default=None,
        help="Second outdir used only for --mode table-ratios: the SECOND group in --table_groups is read from here. "
             "If omitted, both groups are read from --outdir."
    )


    p.add_argument("--lrs", type=float, nargs="+", default=[5e-3, 1e-2, 5e-2, 1e-1])

    p.add_argument("--no_gd", action="store_true")
    p.add_argument("--no_muon", action="store_true")
    p.add_argument("--no_adam", action="store_true")

    p.add_argument("--num_experiments", type=int, default=1)

    p.add_argument(
        "--kinds", nargs="+",
        default=[
            "uniform", "flat_max", "powerlaw", "flat_min", "geometric",
            "geometric_0.3", "geometric_0.9",
            "linear_decay_to_smax", "linear_decay_faster",
            "spikes_2_0.99_0.01", "spikes_2_0.95_0.05", "spikes_2_0.9_0.1",
        ],
    )

    # plot-mode options
    p.add_argument("--match", type=str, default=None, help="substring filter on instance dir basename")
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--max_experiments", type=int, default=None)
    p.add_argument("--plot_outdir", type=str, default=None, help="if set, write plots here instead of inside each instance")
    p.add_argument("--hide_grad_norm", action="store_true")
    p.add_argument("--hide_grad_cond", action="store_true")
    p.add_argument("--no_show", action="store_true", help="do not display figures (recommended for batch); still saves PNGs")

    p.add_argument("--instance_dir", type=str, default=None,
               help="Inspect a specific instance directory (overrides --match scanning).")
    p.add_argument("--inspect_experiments", type=int, default=1,
               help="How many exp_*.pt files to load per instance (default: 1).")

    p.add_argument("--plot_separate", action="store_true", default=False)

    # table options.
    p.add_argument("--table_algos", nargs="+", default=None,
               help="Algo selectors (exact key, substring, or regex via re:...). If omitted, uses all algos found.")
    p.add_argument("--table_ci", type=float, default=1.96,
               help="Multiplier for SE in printed mean ± ci*SE (default 1.96).")
    p.add_argument("--table_out_json", type=str, default=None,
               help="If set, save computed table stats to this JSON file.")

    # table-mode options (best-lr per W0 inside each group)
    p.add_argument(
        "--table_groups", nargs="+", default=None,
        help=(
            "Algorithm groups as NAME=PATTERN. PATTERN matches saved result keys; "
            "can be substring (default) or regex via re:.... "
            "Example: GD=GD_lr_ MuonExactNoMom=Muon_exact_nest_mom0_"
        ),
    )

    # table-ratios numerical stability knobs
    p.add_argument("--ratio_eps", type=float, default=1e-300,
               help="Additive epsilon inside logs for table-ratios: log(x+eps).")
    p.add_argument("--ratio_tol", type=float, default=1e-15,
               help="Tie tolerance in log-space for table-ratios. "
                    "A wins if log_ratio < -tol, B wins if log_ratio > +tol.")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    seed_everything(args.seed)

    print(args.kinds)

    if args.mode == "sweep":
        run_sweep(args, device)
    elif args.mode == "plot":
        run_plot(args)
    elif args.mode == "table":
        run_table(args)
    elif args.mode == "table-ratios":
        run_table_ratios(args)





if __name__ == "__main__":
    main()
