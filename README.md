
### Figure 4 (Section 4.2): median hitting time vs noise

- `fig4_hitting_time_vs_noise.ipynb`: Reproduces Figure 4 (Section 4.2).  
  It simulates the 1D stochastic recursion
  $$s_{t+1} = s_t - \alpha \left(\mathrm{sign}(s_t) + \sigma\xi_t\right)$$ with i.i.d. $\xi_t \sim \mathcal N(0,1)$, 
  and plots the median hitting time (with 95\% central interval) to reach
  $|s_t| \le \varepsilon$ as a function of the noise std $\sigma$.

### Figure 5 (Section 5) and Figure 16 (Appendix G): quadratic exact line search (GD vs Greedy)

- `fig5_quadratic_exact_linesearch.py`: Reproduces Figure 5 (Section 5) and Figure 16 (Appendix G). 
  We minimize a quadratic $$L(W)=\tfrac12\langle W, A W\rangle$$ 
  using exact line search along either the gradient direction $\nabla L(W)$ or the Muon/polar direction
  $\mathrm{polar}(\nabla L(W))$. The Greedy policy chooses the direction with the largest decrease in the loss at each iteration. 
  We plot the median trajectories for GD and Greedy (with 95\% central bands over random initializations).


### Other figures
  See `quadratic_experiments/run_experiments.sh` for details about running experiments and generating figures and tables.