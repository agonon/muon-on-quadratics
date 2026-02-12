# pretty_eigs_1x8.py
import os
import argparse
import numpy as np
import torch


# Pretty display names + optional filtering (set to None to drop)
KIND_RENAME = {
    "flat_max": "max_spiked",
    "flat_min": "min_spiked",
    "gaussian": "gaussian",
    "geometric_0.9": "geometric_decay_to_max",
    "linear_decay_to_smax": "linear_decay_to_max",
    "linear_decay_faster": "linear_decay_to_max",
    "u_shaped_strong": "u_shaped",
    "u_shaped_weak": None, 
    "uniform": "uniform",
}


def find_instance_dirs(runs_dir: str, match=None):
    inst_dirs = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if not os.path.isdir(d):
            continue
        if match and (match not in name):
            continue
        if os.path.isfile(os.path.join(d, "problem.pt")):
            inst_dirs.append(d)
    return inst_dirs


def load_eigs(problem: dict) -> np.ndarray:
    if "A_evals" in problem and torch.is_tensor(problem["A_evals"]):
        return problem["A_evals"].detach().cpu().double().flatten().numpy()
    if "A" not in problem or not torch.is_tensor(problem["A"]):
        raise KeyError("problem.pt missing 'A' and 'A_evals'.")
    A = problem["A"].detach().cpu().double()
    A_sym = 0.5 * (A + A.T)
    return torch.linalg.eigvalsh(A_sym).flatten().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--match", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="default: <runs_dir>/eigs_overview_1x8.html")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--log10", action="store_true", help="plot log10(eigenvalues); requires positive eigs")
    p.add_argument("--share_x", action="store_true", help="use global x-range for all panels")
    p.add_argument("--template", type=str, default="simple_white")
    p.add_argument("--write_png", action="store_true", help="also write PNG (needs kaleido)")
    args = p.parse_args()

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("Install plotly: pip install plotly\n(optional PNG) pip install kaleido")

    if not os.path.isdir(args.runs_dir):
        raise FileNotFoundError(f"runs_dir not found: {args.runs_dir}")

    inst_dirs = find_instance_dirs(args.runs_dir, args.match)
    if len(inst_dirs) == 0:
        raise SystemExit("No instance dirs with problem.pt found")

    # Load up to 8 valid panels (keep scanning even if we drop some kinds)
    panels = []  # each: (display_kind, base_dirname, ev_raw, x_plotted)
    for d in inst_dirs:
        prob = torch.load(os.path.join(d, "problem.pt"), map_location="cpu")

        raw_kind = str(prob.get("kind", os.path.basename(d)))
        display_kind = KIND_RENAME.get(raw_kind, raw_kind)  # keep original if not listed
        if display_kind is None:
            continue  # drop u_shaped_weak

        base = os.path.basename(d)

        ev = load_eigs(prob)
        ev = ev[np.isfinite(ev)]
        if ev.size == 0:
            continue

        x = ev.copy()
        if args.log10:
            x = x[x > 0]
            if x.size == 0:
                continue
            x = np.log10(x)

        panels.append((display_kind, base, ev, x))
        if len(panels) == 8:
            break

    if len(panels) == 0:
        raise SystemExit("After filtering/renaming/log10, no panels left to plot")

    if len(panels) < 8:
        print(f"[warn] only {len(panels)} panels after filtering; leaving {8-len(panels)} empty subplot(s)")

    # Global x-range if requested (based on plotted x)
    if args.share_x:
        gmin = min(float(x.min()) for (_, _, _, x) in panels)
        gmax = max(float(x.max()) for (_, _, _, x) in panels)
        pad = 0.03 * (gmax - gmin + 1e-12)
        x_range = [gmin - pad, gmax + pad]
    else:
        x_range = None

    # ====== SINGLE ROW LAYOUT (1×8) ======
    rows, cols = 1, len(panels)

    # Always create 8 subplot titles (pad with blanks)
    titles = [k for (k, _, _, _) in panels] + [""] * (8 - len(panels))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.035,
    )

    x_label = "eigenvalue" if not args.log10 else "log10(eigenvalue)"
    y_label = "count"

    # Add traces for the panels we have
    for i, (kind, base, ev_raw, x) in enumerate(panels):
        r = 1
        c = i + 1  # 1..8

        ev_min = float(ev_raw.min())
        ev_max = float(ev_raw.max())
        kappa = (ev_max / ev_min) if (ev_min > 0.0 and ev_max > 0.0) else float("inf")

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=x,
                nbinsx=int(args.bins),
                opacity=0.78,
                marker=dict(line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"{x_label}=%{{x:.4g}}<br>{y_label}=%{{y}}<extra></extra>",
            ),
            row=r, col=c,
        )

        # Rug plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.zeros_like(x),
                mode="markers",
                marker=dict(size=6, symbol="line-ns-open"),
                opacity=0.55,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r, col=c,
        )

        # Small panel annotation (top-left)
        ax_id = "" if (i == 0) else str(i + 1)
        fig.add_annotation(
            text=f" ",
            xref=f"x{ax_id} domain",
            yref=f"y{ax_id} domain",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=11),
            opacity=0.9,
        )

        if x_range is not None:
            fig.update_xaxes(range=x_range, row=r, col=c)

    fig.update_layout(
        template=args.template,
        height=420,
        width=2200,
        bargap=0.06,
        margin=dict(l=70, r=20, t=90, b=70),
    )

    # Clean outer labels only
    for rr in range(1, rows + 1):
        for cc in range(1, cols + 1):
            # With a single row, this puts x-labels on all panels (since it's the "bottom" row).
            fig.update_xaxes(title_text=(x_label if rr == rows else ""), row=rr, col=cc)
            fig.update_yaxes(title_text=(y_label if cc == 1 else ""), row=rr, col=cc)

    out_html = args.out or os.path.join(args.runs_dir, "eigs_overview_1x8.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote: {out_html}")

    if args.write_png:
        try:
            out_png = os.path.splitext(out_html)[0] + ".pdf"
            fig.write_image(out_png, scale=2, format="pdf")
            print(f"Wrote: {out_png}")
        except Exception as e:
            print("PNG export failed. Install kaleido: pip install kaleido")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
