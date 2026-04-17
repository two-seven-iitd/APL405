"""
Derived Quantities Plotter — Standalone script.

Loads saved model weights and generates:
  1. Shear force V(x) comparison (predicted vs analytical)
  2. Bending stress contour sigma(x, y) — predicted and analytical side by side
  3. Bending strain contour epsilon(x, y) — predicted and analytical side by side
  4. Max bending stress sigma_max(x) along the beam

Works independently — does NOT import from the main project.
Only reads .pth weight files and uses beam parameters directly.

Usage:
    python derived_quantities/generate_plots.py --beam cantilever
    python derived_quantities/generate_plots.py --beam simply_supported
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════════════
# Beam parameters (duplicated here for independence)
# ═══════════════════════════════════════════════════════════════════════════════
L   = 1.0          # Beam length [m]
E   = 210e9        # Young's modulus [Pa]
I   = 8.33e-6      # Moment of inertia [m^4]
EI  = 1749.3       # Flexural rigidity [N-m^2]
q   = 1000.0       # Distributed load [N/m]

# Cross-section (rectangular): I = b*h^3/12  =>  b=0.01, h=0.1
b   = 0.01         # Width [m]
h   = 0.1          # Height [m]

# Network architecture
W_HIDDEN = 8
M_HIDDEN = 5
NEURONS  = 64


# ═══════════════════════════════════════════════════════════════════════════════
# Network definitions (self-contained, no imports from main project)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_network(n_hidden, n_neurons):
    layers = []
    in_f = 1
    for _ in range(n_hidden):
        lin = nn.Linear(in_f, n_neurons)
        layers.append(lin)
        layers.append(nn.SiLU())
        in_f = n_neurons
    layers.append(nn.Linear(in_f, 1))
    return nn.Sequential(*layers)


class WNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _build_network(W_HIDDEN, NEURONS)
    def forward(self, x):
        return self.net(x)


class MNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _build_network(M_HIDDEN, NEURONS)
    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical solutions
# ═══════════════════════════════════════════════════════════════════════════════
def analytical_M_cantilever(x):
    return -(q * L**2) / 2 + q * L * x - (q * x**2) / 2

def analytical_V_cantilever(x):
    return q * L - q * x

def analytical_M_simply_supported(x):
    return (q * x) / 2 * (L - x)

def analytical_V_simply_supported(x):
    return q * L / 2 - q * x


# ═══════════════════════════════════════════════════════════════════════════════
# Autodiff helpers
# ═══════════════════════════════════════════════════════════════════════════════
def grad1(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Load models
# ═══════════════════════════════════════════════════════════════════════════════
def load_models(beam):
    models_dir = os.path.join("results", beam, "models")
    if not os.path.isdir(models_dir):
        print(f"Error: {models_dir} not found. Train the {beam} model first.")
        sys.exit(1)

    w_net = WNet()
    m_net = MNet()
    w_net.load_state_dict(torch.load(
        os.path.join(models_dir, "hybrid_w.pth"), map_location="cpu", weights_only=True
    ))
    m_net.load_state_dict(torch.load(
        os.path.join(models_dir, "hybrid_m.pth"), map_location="cpu", weights_only=True
    ))
    w_net.eval()
    m_net.eval()
    return w_net, m_net


# ═══════════════════════════════════════════════════════════════════════════════
# Compute predicted quantities
# ═══════════════════════════════════════════════════════════════════════════════
def compute_predicted(w_net, m_net, x_np):
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    # M(x) and V(x) = dM/dx
    M_pred = m_net(x)
    V_pred = grad1(M_pred, x)

    M_np = M_pred.detach().squeeze().numpy()
    V_np = V_pred.detach().squeeze().numpy()

    return M_np, V_np


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Shear force V(x)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_shear_force(x_np, V_pred, V_ana, beam, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, V_ana,  "k--", lw=2.5, label="Analytical", alpha=0.8)
    ax.plot(x_np, V_pred, "b-",  lw=1.8, label="Hybrid PINN")
    ax.set_xlabel("x  [m]", fontsize=12)
    ax.set_ylabel("V(x)  [N]", fontsize=12)
    ax.set_title(f"Shear Force Diagram  ---  {beam.replace('_', ' ').title()} Beam",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "shear_force.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Max bending stress sigma_max(x) = M(x) * (h/2) / I
# ═══════════════════════════════════════════════════════════════════════════════
def plot_max_stress(x_np, M_pred, M_ana, beam, out_dir):
    y_max = h / 2
    sigma_pred = M_pred * y_max / I
    sigma_ana  = M_ana  * y_max / I

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_np, sigma_ana / 1e6,  "k--", lw=2.5, label="Analytical", alpha=0.8)
    ax.plot(x_np, sigma_pred / 1e6, "r-",  lw=1.8, label="Hybrid PINN")
    ax.set_xlabel("x  [m]", fontsize=12)
    ax.set_ylabel(r"$\sigma_{max}(x)$  [MPa]", fontsize=12)
    ax.set_title(f"Maximum Bending Stress  ---  {beam.replace('_', ' ').title()} Beam",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "max_bending_stress.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 & 4: Stress and strain contour plots (x vs y)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stress_contour(x_np, M_pred, M_ana, beam, out_dir):
    """Bending stress sigma(x, y) = M(x) * y / I"""
    ny = 100
    y_arr = np.linspace(-h/2, h/2, ny)
    X, Y = np.meshgrid(x_np, y_arr)

    # Predicted: sigma(x,y) = M_pred(x) * y / I
    sigma_pred = np.outer(y_arr, M_pred) / I   # shape (ny, nx)
    sigma_ana  = np.outer(y_arr, M_ana)  / I

    # Convert to MPa
    sigma_pred_MPa = sigma_pred / 1e6
    sigma_ana_MPa  = sigma_ana  / 1e6

    vmin = min(sigma_pred_MPa.min(), sigma_ana_MPa.min())
    vmax = max(sigma_pred_MPa.max(), sigma_ana_MPa.max())

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  hspace=0.35, wspace=0.25)

    # Top left: Analytical
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, Y * 1000, sigma_ana_MPa, levels=50,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_title("Analytical", fontsize=12, fontweight="bold")
    ax1.set_ylabel("y  [mm]", fontsize=11)
    ax1.set_xlabel("x  [m]", fontsize=11)
    ax1.tick_params(labelsize=9)
    plt.colorbar(c1, ax=ax1, label=r"$\sigma$  [MPa]", pad=0.02)

    # Top right: Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.contourf(X, Y * 1000, sigma_pred_MPa, levels=50,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax2.set_title("Hybrid PINN", fontsize=12, fontweight="bold")
    ax2.set_ylabel("y  [mm]", fontsize=11)
    ax2.set_xlabel("x  [m]", fontsize=11)
    ax2.tick_params(labelsize=9)
    plt.colorbar(c2, ax=ax2, label=r"$\sigma$  [MPa]", pad=0.02)

    # Bottom left: Absolute error
    err_MPa = np.abs(sigma_pred_MPa - sigma_ana_MPa)
    ax3 = fig.add_subplot(gs[1, 0])
    c3 = ax3.contourf(X, Y * 1000, err_MPa, levels=50, cmap="hot_r")
    ax3.set_title("Absolute Error", fontsize=12, fontweight="bold")
    ax3.set_ylabel("y  [mm]", fontsize=11)
    ax3.set_xlabel("x  [m]", fontsize=11)
    ax3.tick_params(labelsize=9)
    plt.colorbar(c3, ax=ax3, label=r"$|\Delta\sigma|$  [MPa]", pad=0.02)

    # Bottom right: Stress profile at critical section
    if beam == "cantilever":
        x_crit_idx = 0           # x=0 (clamped end, max moment)
        x_crit_label = "x = 0 (clamped end)"
    else:
        x_crit_idx = len(x_np) // 2  # midspan
        x_crit_label = "x = L/2 (midspan)"

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(sigma_ana_MPa[:, x_crit_idx],  y_arr * 1000, "k--", lw=2.5,
             label="Analytical", alpha=0.8)
    ax4.plot(sigma_pred_MPa[:, x_crit_idx], y_arr * 1000, "r-",  lw=1.8,
             label="Hybrid PINN")
    ax4.set_xlabel(r"$\sigma$  [MPa]", fontsize=11)
    ax4.set_ylabel("y  [mm]", fontsize=11)
    ax4.set_title(f"Stress Profile at {x_crit_label}", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=9)

    fig.suptitle(f"Bending Stress  " + r"$\sigma(x, y) = M(x) \cdot y \,/\, I$"
                 + f"  ---  {beam.replace('_', ' ').title()} Beam",
                 fontsize=14, fontweight="bold", y=1.01)
    path = os.path.join(out_dir, "stress_contour.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_strain_contour(x_np, M_pred, M_ana, beam, out_dir):
    """Bending strain epsilon(x, y) = sigma(x, y) / E = M(x) * y / (E * I)"""
    ny = 100
    y_arr = np.linspace(-h/2, h/2, ny)
    X, Y = np.meshgrid(x_np, y_arr)

    eps_pred = np.outer(y_arr, M_pred) / (E * I)
    eps_ana  = np.outer(y_arr, M_ana)  / (E * I)

    # Convert to microstrain
    eps_pred_us = eps_pred * 1e6
    eps_ana_us  = eps_ana  * 1e6

    vmin = min(eps_pred_us.min(), eps_ana_us.min())
    vmax = max(eps_pred_us.max(), eps_ana_us.max())

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X, Y * 1000, eps_ana_us, levels=50,
                       cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax1.set_title("Analytical", fontsize=12, fontweight="bold")
    ax1.set_ylabel("y  [mm]", fontsize=11)
    ax1.set_xlabel("x  [m]", fontsize=11)
    ax1.tick_params(labelsize=9)
    plt.colorbar(c1, ax=ax1, label=r"$\varepsilon$  [$\mu\varepsilon$]", pad=0.02)

    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.contourf(X, Y * 1000, eps_pred_us, levels=50,
                       cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax2.set_title("Hybrid PINN", fontsize=12, fontweight="bold")
    ax2.set_ylabel("y  [mm]", fontsize=11)
    ax2.set_xlabel("x  [m]", fontsize=11)
    ax2.tick_params(labelsize=9)
    plt.colorbar(c2, ax=ax2, label=r"$\varepsilon$  [$\mu\varepsilon$]", pad=0.02)

    err_us = np.abs(eps_pred_us - eps_ana_us)
    ax3 = fig.add_subplot(gs[1, 0])
    c3 = ax3.contourf(X, Y * 1000, err_us, levels=50, cmap="hot_r")
    ax3.set_title("Absolute Error", fontsize=12, fontweight="bold")
    ax3.set_ylabel("y  [mm]", fontsize=11)
    ax3.set_xlabel("x  [m]", fontsize=11)
    ax3.tick_params(labelsize=9)
    plt.colorbar(c3, ax=ax3, label=r"$|\Delta\varepsilon|$  [$\mu\varepsilon$]", pad=0.02)

    # Strain profile at critical section
    if beam == "cantilever":
        x_crit_idx = 0
        x_crit_label = "x = 0 (clamped end)"
    else:
        x_crit_idx = len(x_np) // 2
        x_crit_label = "x = L/2 (midspan)"

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(eps_ana_us[:, x_crit_idx],  y_arr * 1000, "k--", lw=2.5,
             label="Analytical", alpha=0.8)
    ax4.plot(eps_pred_us[:, x_crit_idx], y_arr * 1000, "b-",  lw=1.8,
             label="Hybrid PINN")
    ax4.set_xlabel(r"$\varepsilon$  [$\mu\varepsilon$]", fontsize=11)
    ax4.set_ylabel("y  [mm]", fontsize=11)
    ax4.set_title(f"Strain Profile at {x_crit_label}", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=9)

    fig.suptitle(r"Bending Strain  $\varepsilon(x, y) = M(x) \cdot y \,/\, (EI)$"
                 + f"  ---  {beam.replace('_', ' ').title()} Beam",
                 fontsize=14, fontweight="bold", y=1.01)
    path = os.path.join(out_dir, "strain_contour.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate derived quantity plots")
    parser.add_argument("--beam", default="cantilever",
                        choices=["cantilever", "simply_supported"])
    args = parser.parse_args()
    beam = args.beam

    out_dir = os.path.join("derived_quantities", "plots", beam)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDerived Quantities — {beam.replace('_', ' ').title()} Beam")
    print(f"Cross-section: rectangular {b*1000:.0f} mm x {h*1000:.0f} mm")
    print(f"Output: {out_dir}/\n")

    # Load models
    w_net, m_net = load_models(beam)

    # Evaluation grid
    nx = 500
    x_np = np.linspace(0.0, 1.0, nx)

    # Compute predicted M and V
    M_pred, V_pred = compute_predicted(w_net, m_net, x_np)

    # Analytical
    if beam == "cantilever":
        M_ana = analytical_M_cantilever(x_np)
        V_ana = analytical_V_cantilever(x_np)
    else:
        M_ana = analytical_M_simply_supported(x_np)
        V_ana = analytical_V_simply_supported(x_np)

    # Generate all plots
    print("Generating plots:")
    plot_shear_force(x_np, V_pred, V_ana, beam, out_dir)
    plot_max_stress(x_np, M_pred, M_ana, beam, out_dir)
    plot_stress_contour(x_np, M_pred, M_ana, beam, out_dir)
    plot_strain_contour(x_np, M_pred, M_ana, beam, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
