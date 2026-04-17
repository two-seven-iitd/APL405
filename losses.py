# losses.py — Derivative utilities and all loss functions

import torch
from config import EI, q, L


# ── Autodiff helpers ──────────────────────────────────────────────────────────

def grad1(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """First derivative d(output)/dx.  create_graph=True is MANDATORY."""
    return torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def grad2(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Second derivative d²(output)/dx²."""
    return grad1(grad1(output, x), x)


# ── Individual loss terms ─────────────────────────────────────────────────────

def loss_equilibrium(M_net, x_coll: torch.Tensor) -> torch.Tensor:
    """L_Ω = mean( (M''(x) + q)² )  over collocation points."""
    M_pred = M_net(x_coll)
    M_xx   = grad2(M_pred, x_coll)
    return torch.mean((M_xx + q) ** 2)


def loss_coupling(w_net, M_net, x_coll: torch.Tensor) -> torch.Tensor:
    """L_Υ = mean( (M(x) - EI·w''(x))² )  over collocation points."""
    w_pred = w_net(x_coll)
    M_pred = M_net(x_coll)
    w_xx   = grad2(w_pred, x_coll)
    return torch.mean((M_pred - EI * w_xx) ** 2)


def loss_bc_cantilever(w_net, M_net, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """L_Γ for cantilever: w(0)=0, w'(0)=0, M(1)=0, M'(1)=0."""
    w_at_0  = w_net(x0)
    w_x_at_0 = grad1(w_at_0, x0)

    M_at_1  = M_net(x1)
    M_x_at_1 = grad1(M_at_1, x1)

    return (
        w_at_0.squeeze() ** 2
        + w_x_at_0.squeeze() ** 2
        + M_at_1.squeeze() ** 2
        + M_x_at_1.squeeze() ** 2
    )


def loss_bc_simply_supported(w_net, M_net, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """L_Γ for simply-supported: w(0)=0, w(1)=0, M(0)=0, M(1)=0."""
    return (
        w_net(x0).squeeze() ** 2
        + w_net(x1).squeeze() ** 2
        + M_net(x0).squeeze() ** 2
        + M_net(x1).squeeze() ** 2
    )


def loss_data(w_net, x_sensors: torch.Tensor, w_measured: torch.Tensor) -> torch.Tensor:
    """L_D = mean( (w_net(x_sensors) - w_measured)² )."""
    w_pred = w_net(x_sensors)
    return torch.mean((w_pred - w_measured) ** 2)


# ── Combined losses for each model ────────────────────────────────────────────

def total_loss_physics_only(
    w_net, M_net, x_coll, x0, x1,
    lam_omega, lam_upsilon, lam_gamma,
    beam: str = "cantilever",
):
    L_omega   = loss_equilibrium(M_net, x_coll)
    L_upsilon = loss_coupling(w_net, M_net, x_coll)
    if beam == "cantilever":
        L_gamma = loss_bc_cantilever(w_net, M_net, x0, x1)
    else:
        L_gamma = loss_bc_simply_supported(w_net, M_net, x0, x1)

    total = lam_omega * L_omega + lam_upsilon * L_upsilon + lam_gamma * L_gamma
    return total, L_omega, L_upsilon, L_gamma


def total_loss_hybrid(
    w_net, M_net, x_coll, x0, x1, x_sensors, w_measured,
    lam_omega, lam_upsilon, lam_gamma, lam_data,
    beam: str = "cantilever",
):
    total, L_omega, L_upsilon, L_gamma = total_loss_physics_only(
        w_net, M_net, x_coll, x0, x1, lam_omega, lam_upsilon, lam_gamma, beam
    )
    L_d  = loss_data(w_net, x_sensors, w_measured)
    total = total + lam_data * L_d
    return total, L_omega, L_upsilon, L_gamma, L_d
