
### Figure 4 (Section 4.2): median hitting time vs noise

- `fig4_hitting_time_vs_noise.ipynb`: Reproduces Figure 4 (Section 4.2).  
  It simulates the 1D stochastic recursion
  \[
  s_{t+1} = s_t - \alpha \left(\operatorname{sign}(s_t) + \sigma\xi_t\right),
  \qquad \xi_t \sim \mathcal N(0,1),
  \]
  and plots the median hitting time (with 95% central interval) to reach
  $|s_t| \le \varepsilon$ as a function of the noise standard deviation $\sigma$.


  ### Figure 5 (Section 5) and Figure 16 (Appendix G): quadratic exact line search (GD vs Greedy)

- `fig5_quadratic_exact_linesearch.py`: Reproduces Figure 5 (Section 5) and Figure 16 (Appendix G).
  We minimize a quadratic
  \[
  L(W)=\tfrac12\langle W, A W\rangle - \langle B, W\rangle
  \quad (A \succ 0),
  \]
  using exact line search along either the gradient direction $D_{\mathrm{GD}}=\nabla L(W)$ or 
  a greedy approach, which selects the best between the gradient and the Muon/polar direction
  $D_{\mu}=\operatorname{polar}(\nabla L(W))$.
  Given direction $D$, the exact line search step size is computed in closed form as
  \[
  \alpha^\star = \frac{\langle D,\nabla L(W)\rangle}{\langle D, A D\rangle}.
  \]
  Plots median trajectories (with 95% central bands over random seeds).