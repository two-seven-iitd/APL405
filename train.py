# train.py — Training loops for all three models

import torch
import torch.optim as optim
from config import (
    EPOCHS, LR, LR_PATIENCE, LR_FACTOR, LOG_EVERY, GRAD_CLIP, RESAMPLE_EVERY,
    N_COLLOC, N_SENSORS, NOISE_PCT, get_loss_weights,
)
from models import WNet, MNet, DataOnlyNet
from data import (
    get_collocation_points, get_sensor_data,
    get_boundary_points,
)
from losses import (
    loss_data,
    total_loss_physics_only,
    total_loss_hybrid,
)


def train_data_only(
    device: str = "cpu",
    epochs: int = EPOCHS,
    beam: str = "cantilever",
    n_sensors: int = N_SENSORS,
    noise_pct: float = NOISE_PCT,
    seed: int = 42,
):
    """Model A: data-only baseline — single network, L_D only."""
    net = DataOnlyNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR
    )

    x_sensors, w_noisy, _ = get_sensor_data(
        n_sensors=n_sensors, noise_pct=noise_pct, beam=beam, seed=seed, device=device
    )

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        L_d = loss_data(net, x_sensors, w_noisy)
        L_d.backward()
        optimizer.step()
        scheduler.step(L_d)

        history.append(L_d.item())
        if epoch % LOG_EVERY == 0:
            print(f"[Data-Only]  epoch {epoch:>6}  L_D={L_d.item():.4e}  lr={optimizer.param_groups[0]['lr']:.2e}")

    return net, history


def train_physics_only(
    device: str = "cpu",
    epochs: int = EPOCHS,
    beam: str = "cantilever",
):
    """Model B: physics-only Mixed-PINN — L_Ω + L_Υ + L_Γ."""
    lw = get_loss_weights(beam)
    lam_omega   = lw["lambda_omega"]
    lam_upsilon = lw["lambda_upsilon"]
    lam_gamma   = lw["lambda_gamma"]

    w_net = WNet().to(device)
    m_net = MNet().to(device)
    all_params = list(w_net.parameters()) + list(m_net.parameters())
    optimizer = optim.Adam(all_params, lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR
    )

    x0, x1 = get_boundary_points(device=device)
    x_coll = get_collocation_points(N_COLLOC, device=device)
    history = {"total": [], "omega": [], "upsilon": [], "gamma": []}

    for epoch in range(1, epochs + 1):
        if RESAMPLE_EVERY > 0 and epoch % RESAMPLE_EVERY == 0:
            x_coll = get_collocation_points(N_COLLOC, device=device)

        optimizer.zero_grad()
        total, L_omega, L_upsilon, L_gamma = total_loss_physics_only(
            w_net, m_net, x_coll, x0, x1,
            lam_omega, lam_upsilon, lam_gamma, beam,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step(total)

        history["total"].append(total.item())
        history["omega"].append(L_omega.item())
        history["upsilon"].append(L_upsilon.item())
        history["gamma"].append(L_gamma.item())

        if epoch % LOG_EVERY == 0:
            print(
                f"[Physics-Only]  epoch {epoch:>6}  total={total.item():.4e}  "
                f"L_Om={L_omega.item():.4e}  L_Up={L_upsilon.item():.4e}  "
                f"L_Ga={L_gamma.item():.4e}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return w_net, m_net, history


def train_hybrid(
    device: str = "cpu",
    epochs: int = EPOCHS,
    beam: str = "cantilever",
    n_sensors: int = N_SENSORS,
    noise_pct: float = NOISE_PCT,
    seed: int = 42,
    warm_start_w=None,
    warm_start_m=None,
):
    """Model C: Hybrid Mixed-PINN — L_Ω + L_Υ + L_Γ + L_D.

    warm_start_w / warm_start_m: optional state dicts to initialise w_net and
    m_net from saved Physics-Only weights instead of random Xavier init.
    """
    lw = get_loss_weights(beam)
    lam_omega   = lw["lambda_omega"]
    lam_upsilon = lw["lambda_upsilon"]
    lam_gamma   = lw["lambda_gamma"]
    lam_data    = lw["lambda_data"]

    w_net = WNet().to(device)
    m_net = MNet().to(device)
    if warm_start_w is not None:
        w_net.load_state_dict(warm_start_w)
    if warm_start_m is not None:
        m_net.load_state_dict(warm_start_m)
    all_params = list(w_net.parameters()) + list(m_net.parameters())
    optimizer = optim.Adam(all_params, lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR
    )

    x_sensors, w_noisy, _ = get_sensor_data(
        n_sensors=n_sensors, noise_pct=noise_pct, beam=beam, seed=seed, device=device
    )
    x0, x1 = get_boundary_points(device=device)
    x_coll = get_collocation_points(N_COLLOC, device=device)
    history = {"total": [], "omega": [], "upsilon": [], "gamma": [], "data": []}

    for epoch in range(1, epochs + 1):
        if RESAMPLE_EVERY > 0 and epoch % RESAMPLE_EVERY == 0:
            x_coll = get_collocation_points(N_COLLOC, device=device)

        optimizer.zero_grad()
        total, L_omega, L_upsilon, L_gamma, L_d = total_loss_hybrid(
            w_net, m_net, x_coll, x0, x1, x_sensors, w_noisy,
            lam_omega, lam_upsilon, lam_gamma, lam_data, beam,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step(total)

        history["total"].append(total.item())
        history["omega"].append(L_omega.item())
        history["upsilon"].append(L_upsilon.item())
        history["gamma"].append(L_gamma.item())
        history["data"].append(L_d.item())

        if epoch % LOG_EVERY == 0:
            print(
                f"[Hybrid]  epoch {epoch:>6}  total={total.item():.4e}  "
                f"L_Om={L_omega.item():.4e}  L_Up={L_upsilon.item():.4e}  "
                f"L_Ga={L_gamma.item():.4e}  L_D={L_d.item():.4e}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return w_net, m_net, history
