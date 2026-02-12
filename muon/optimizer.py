# muon/optimizer.py
import torch
from .projections import newtonschulz5, polar_exact


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
        projection="exact",
        ns_steps=5,
        eps=1e-7,
    ):
        if projection not in ("exact", "ns"):
            raise ValueError("projection must be 'exact' or 'ns'")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0,1), got {momentum}")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                projection=projection,
                ns_steps=ns_steps,
                eps=eps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            proj = group["projection"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon assumes 2D params only, got {tuple(p.shape)}")

                g = p.grad

                # --- EMA momentum buffer ---
                if mu != 0.0:
                    st = self.state.setdefault(p, {})
                    buf = st.get("buf")
                    if buf is None:
                        buf = torch.zeros_like(g)

                    # buf_t = mu*buf_{t-1} + (1-mu)*g_t
                    buf.mul_(mu).add_(g, alpha=(1.0 - mu))
                    st["buf"] = buf

                    # --- choose which "momentum variant" to project ---
                    if nesterov:
                        # g_used = (1-mu)*g + mu*buf
                        g_used = g.mul(1.0 - mu).add(buf, alpha=mu)
                    else:
                        # g_used = g + (1-mu)*buf
                        g_used = g.add(buf, alpha=(1.0 - mu))
                else:
                    g_used = g

                # --- Project to polar factor ---
                if proj == "exact":
                    Q = polar_exact(g_used)
                else:
                    Q = newtonschulz5(g_used, steps=ns_steps, eps=eps).to(dtype=p.dtype)

                # --- Update ---
                p.add_(Q, alpha=-lr)

        return loss
