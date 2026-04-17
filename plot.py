# plot.py — All visualisation

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (
    get_eval_grid,
    get_sensor_data,
    analytical_w_cantilever,
    analytical_M_cantilever,
    analytical_w_simply_supported,
    analytical_M_simply_supported,
)


def _eval(net, device, n=1000):
    x = get_eval_grid(n=n, device=device)
    with torch.no_grad():
        y = net(x).squeeze().cpu().numpy()
    return y


def plot_deflection_comparison(
    data_only_net,
    physics_w_net,
    hybrid_w_net,
    plots_dir: str,
    beam: str = "cantilever",
    device: str = "cpu",
    n_sensors: int = 15,
    noise_pct: float = 0.10,
    fname: str = "deflection_comparison.png",
):
    """Plot 1: Deflection comparison across all three models."""
    x_np = np.linspace(0.0, 1.0, 1000)
    w_ana = (
        analytical_w_cantilever(x_np)
        if beam == "cantilever"
        else analytical_w_simply_supported(x_np)
    )

    w_data    = _eval(data_only_net, device)
    w_physics = _eval(physics_w_net, device)
    w_hybrid  = _eval(hybrid_w_net,  device)

    x_s, w_noisy, _ = get_sensor_data(
        n_sensors=n_sensors, noise_pct=noise_pct, beam=beam, device=device
    )
    x_s_np   = x_s.cpu().numpy().squeeze()
    w_s_np   = w_noisy.cpu().numpy().squeeze()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_np, w_ana,     "k--", lw=2,   label="Analytical")
    ax.plot(x_np, w_data,    "r-",  lw=1.5, label="Data-Only")
    ax.plot(x_np, w_physics, "b-",  lw=1.5, label="Physics-Only")
    ax.plot(x_np, w_hybrid,  "g-",  lw=1.5, label="Hybrid")
    ax.scatter(x_s_np, w_s_np, c="red", s=40, zorder=5, label="Sensors (noisy)")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("w(x)  [m]")
    ax.set_title(f"Deflection Comparison — {beam.title()} Beam")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_moment_comparison(
    physics_m_net,
    hybrid_m_net,
    plots_dir: str,
    beam: str = "cantilever",
    device: str = "cpu",
    fname: str = "moment_comparison.png",
):
    """Plot 2: Bending moment comparison (no data-only M, it has no M-net)."""
    x_np = np.linspace(0.0, 1.0, 1000)
    M_ana = (
        analytical_M_cantilever(x_np)
        if beam == "cantilever"
        else analytical_M_simply_supported(x_np)
    )

    M_physics = _eval(physics_m_net, device)
    M_hybrid  = _eval(hybrid_m_net,  device)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_np, M_ana,      "k--", lw=2,   label="Analytical")
    ax.plot(x_np, M_physics,  "b-",  lw=1.5, label="Physics-Only")
    ax.plot(x_np, M_hybrid,   "g-",  lw=1.5, label="Hybrid")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("M(x)  [N·m]")
    ax.set_title(f"Bending Moment Comparison — {beam.title()} Beam")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pointwise_error(
    data_only_net,
    physics_w_net,
    hybrid_w_net,
    plots_dir: str,
    beam: str = "cantilever",
    device: str = "cpu",
    fname: str = "pointwise_error.png",
):
    """Plot 3: |w_pred - w_analytical| for all three models."""
    x_np = np.linspace(0.0, 1.0, 1000)
    w_ana = (
        analytical_w_cantilever(x_np)
        if beam == "cantilever"
        else analytical_w_simply_supported(x_np)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for net, label, color in [
        (data_only_net, "Data-Only",    "r"),
        (physics_w_net, "Physics-Only", "b"),
        (hybrid_w_net,  "Hybrid",       "g"),
    ]:
        err = np.abs(_eval(net, device) - w_ana)
        ax.plot(x_np, err, color=color, lw=1.5, label=label)

    ax.set_xlabel("x  [m]")
    ax.set_ylabel("|w_pred - w_analytical|  [m]")
    ax.set_title(f"Pointwise Deflection Error — {beam.title()} Beam")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_convergence(
    history_data: list,
    history_physics: dict,
    history_hybrid: dict,
    plots_dir: str,
    fname: str = "convergence.png",
):
    """Plot 4: Total loss vs epoch for all three models."""
    epochs = np.arange(1, len(history_data) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, history_data,              "r-", lw=1.5, label="Data-Only")
    ax.semilogy(epochs, history_physics["total"],  "b-", lw=1.5, label="Physics-Only")
    ax.semilogy(epochs, history_hybrid["total"],   "g-", lw=1.5, label="Hybrid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_individual_losses(
    history_hybrid: dict,
    plots_dir: str,
    fname: str = "individual_losses_hybrid.png",
):
    """Plot 5: Individual loss components for the hybrid model."""
    epochs = np.arange(1, len(history_hybrid["total"]) + 1)
    keys   = [("omega", "L_Ω  (Equilibrium)", "C0"),
              ("upsilon", "L_Υ  (Coupling)",   "C1"),
              ("gamma",   "L_Γ  (Boundary)",   "C2"),
              ("data",    "L_D  (Data)",        "C3")]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in keys:
        ax.semilogy(epochs, history_hybrid[key], color=color, lw=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Hybrid Model — Individual Loss Components")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def save_all_plots(
    data_only_net,
    physics_w_net,
    physics_m_net,
    hybrid_w_net,
    hybrid_m_net,
    history_data,
    history_physics,
    history_hybrid,
    plots_dir: str,
    beam: str = "cantilever",
    device: str = "cpu",
):
    """Convenience: generate all five required plots."""
    os.makedirs(plots_dir, exist_ok=True)
    plot_deflection_comparison(
        data_only_net, physics_w_net, hybrid_w_net,
        plots_dir, beam=beam, device=device,
    )
    plot_moment_comparison(
        physics_m_net, hybrid_m_net,
        plots_dir, beam=beam, device=device,
    )
    plot_pointwise_error(
        data_only_net, physics_w_net, hybrid_w_net,
        plots_dir, beam=beam, device=device,
    )
    plot_convergence(history_data, history_physics, history_hybrid, plots_dir)
    plot_individual_losses(history_hybrid, plots_dir)
