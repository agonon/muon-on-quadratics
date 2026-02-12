# muon/runner.py
import math
import torch

def run(loss_fn, params0, opt_ctor, steps: int, sched_ctor=None):
    """
    loss_fn(params)-> scalar loss tensor
    params0: list of tensors (initial values) or nn.Parameters
    opt_ctor(params)-> torch.optim.Optimizer
    sched_ctor(opt)-> torch.optim.lr_scheduler (optional)
    """
    params = [torch.nn.Parameter(p.detach().clone()) for p in params0]
    opt = opt_ctor(params)
    sch = sched_ctor(opt) if sched_ctor else None

    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(params)
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def compare(loss_fn, params0, algos: dict, steps: int, sched_ctor=None):
    """
    algos: dict name -> opt_ctor(params)
    returns dict name -> list(loss)
    """
    out = {}
    for name, opt_ctor in algos.items():
        out[name] = run(loss_fn, params0, opt_ctor, steps, sched_ctor=sched_ctor)
    return out


def grad_fro_norm(params):
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        s += float((p.grad.detach() ** 2).sum().item())
    return math.sqrt(s)


def condition_number(params):
    """
    Average cond number across parameter grads (2D matrices).
    Returns float.
    """
    vals = []
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.ndim != 2:
            continue
        # cond returns tensor scalar
        vals.append(float(torch.linalg.cond(g).item()))
    return float(sum(vals) / max(1, len(vals)))


def run_trace(loss_fn, params0, opt_ctor, steps: int, sched_ctor=None, W_star=None, UF=None, UC=None):
    """
    If W_star, UF, UC are provided, logs:
      err_F[t] = || UF (W_t - W_star) ||_F
      err_C[t] = || UC (W_t - W_star) ||_F
    """
    params = [torch.nn.Parameter(p.detach().clone()) for p in params0]
    opt = opt_ctor(params)
    sch = sched_ctor(opt) if sched_ctor else None

    lr0 = float(opt.param_groups[0]["lr"])
    sched_name = sch.__class__.__name__ if sch is not None else "None"

    losses, gnorms, gcondnums = [], [], []
    errF, errC = [], []

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(params)
        loss.backward()

        losses.append(float(loss.detach().cpu().item()))
        gnorms.append(grad_fro_norm(params))
        gcondnums.append(condition_number(params))

        # log subspace errors BEFORE the step (consistent with loss/grad)
        if (W_star is not None) and (UF is not None) and (UC is not None):
            W = params[0].detach()
            E = W - W_star
            errF.append(float(torch.linalg.norm(UF @ E).item()))
            errC.append(float(torch.linalg.norm(UC @ E).item()))

        opt.step()
        if sch:
            sch.step()

    out = {
        "loss": losses,
        "grad_norm": gnorms,
        "lr0": lr0,
        "sched": sched_name,
        "grad_cond_num": gcondnums,
    }
    if errF:
        out["err_F"] = errF
        out["err_C"] = errC
    return out


def compare_traces(loss_fn, params0, algos: dict, steps: int, sched_ctor=None, **kwargs):
    return {name: run_trace(loss_fn, params0, opt_ctor, steps, sched_ctor, **kwargs)
            for name, opt_ctor in algos.items()}
