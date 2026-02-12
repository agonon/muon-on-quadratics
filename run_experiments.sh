#!/usr/bin/env bash

# Run experiments for selected spectrum types.
# By default, this runs GD, Adam and Muon. (Can pass --no_gd, --no_muon, --no_adam if needed.)
# To see which specific versions of Muon, check out the make_algo_specs function in main_experiments.py
# (small dimensions here for easy testing).
# For the paper, we used n = d_in = d_out = 128, steps = 500, num_experiments = 100 (takes ~12h on a GeForce RTX 3070 8GB gpu). 
python main_experiments.py \
  --mode sweep \
  --kinds flat_min flat_max uniform gaussian geometric_0.9 linear_decay_to_smax \
  --outdir runs \
  --seed 123 \
  --n 16 \
  --d_in 16 \
  --d_out 16 \
  --steps 100 \
  --s_min 1e-4 \
  --s_max 10.0 \
  --alpha 1.0 \
  --num_experiments 10 \
  --lrs 0.001 0.01 0.1


# Plot one histogram for each spectrum. (Figure 1).
python3 plot_A_eigs_row.py


# Make separate plots (just loss curve) for max_experiments trajectories for each spectrum type. (eg: Figures 10, 11, 12, 13)
# Remove --hide_grad_norm to also plot the gradient norms along the trajectory.
# Here, we only plot one trajectory per spectrum type.
python main_experiments.py \
    --mode plot \
    --outdir runs \
    --max_experiments 1 \
    --kinds flat_min flat_max uniform gaussian geometric_0.9 linear_decay_to_smax  \
    --plot_separate \
    --hide_grad_norm \
    --no_show 

# Make plots with 'median' trajectories (based on max_experiments trajectories) for each spectrum type. (eg: Figures 14, 15)
python main_experiments.py \
    --mode plot \
    --outdir runs \
    --max_experiments 10 \
    --hide_grad_norm \
    --no_show 


# Make tables reporting the "win rate" between two algorithms. (eg: Tables 1, 2 and similar)
# Here, "win" means that the smallest loss achieved by one of the algorithms (across the whole trajectory
# up to time T/10, T/2, T) is better than the other.
# Pick one of the following versions of Muon to compare against GD.
# Muon=[Muon_exact_nest_mom0_ | Muon_exact_nest_mom0.9_ | Muon_ns5_nest_mom0 | Muon_ns5_nest_mom0.9]
python main_experiments.py \
  --mode table-ratios \
  --outdir runs \
  --steps 100 \
  --table_groups Muon=Muon_exact_nest_mom0.9_ GD=GD_lr_ \
  --kinds flat_min flat_max uniform gaussian geometric_0.9 linear_decay_to_smax


# Make bar plot showing how orders of magnitude of loss decrease for GD vs Muon. (eg: Figures 2, 7, 8, 9)
# By default, it's aligned at the top. If --absolute_range is included, then the actual initial loss / finall loss are displayed.
python plot_improvement_bars.py \
    --outdir runs \
    --no_show


# Make a table showing the mean \pm 1.96 SE for best loss for specified GD and Muon runs (not included in paper). 
python main_experiments.py \
  --mode table \
  --outdir runs \
  --steps 100 \
  --kinds flat_min flat_max uniform gaussian geometric_0.9 linear_decay_to_smax \
  --table_groups GD=GD_lr_ MuonExactNoMom=Muon_exact_nest_mom0_ \
  --table_out_json runs_vanishing/table_bestlr.json


# Make tables reporting "win rate" between two algorithms *in different directories* (eg: to compare constant LR with vanishing LR runs).
# python main_experiments.py \
#   --mode table-ratios \
#   --outdir runs_vanishing \
#   --outdir_b runs \
#   --steps 500 \
#   --table_groups MuonInexactWithMomentumVanishingLR=Muon_ns5_nest_mom0.9_ GD=GD_lr_ \
#   --kinds flat_max flat_min uniform gaussian linear_decay_to_smax u_shaped_strong geometric_0.9