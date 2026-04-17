# data.py — Collocation points, analytical solutions, and sensor data

import torch
import numpy as np
from config import L, EI, q, N_COLLOC, N_SENSORS, NOISE_PCT


# ── Analytical solutions ───────────────────────────────────────────────────────

def analytical_w_cantilever(x: np.ndarray) -> np.ndarray:
    """Exact deflection for cantilever beam under uniform load q (clamped at x=0).
    Sign: downward deflection is negative."""
    return -(q * x**2) / (24 * EI) * (6*L**2 - 4*L*x + x**2)


def analytical_M_cantilever(x: np.ndarray) -> np.ndarray:
    """Exact bending moment for cantilever under uniform load q."""
    return -(q * L**2) / 2 + q * L * x - (q * x**2) / 2


def analytical_V_cantilever(x: np.ndarray) -> np.ndarray:
    """Exact shear force for cantilever (consistency check)."""
    return q * L - q * x


def analytical_w_simply_supported(x: np.ndarray) -> np.ndarray:
    """Exact deflection for simply-supported beam under uniform load q."""
    return -(q * x) / (24 * EI) * (L**3 - 2*L*x**2 + x**3)


def analytical_M_simply_supported(x: np.ndarray) -> np.ndarray:
    """Exact bending moment for simply-supported beam under uniform load q."""
    return (q * x) / 2 * (L - x)


# ── Collocation points ─────────────────────────────────────────────────────────

def get_collocation_points(n: int = N_COLLOC, device: str = "cpu") -> torch.Tensor:
    """Return n uniformly spaced collocation points in [0, 1], shape (n, 1).
    x is normalised: x_norm = x / L  (L=1 so identical here).
    requires_grad=True is MANDATORY for autograd."""
    x = torch.linspace(0.0, 1.0, n, device=device).unsqueeze(1)
    x.requires_grad_(True)
    return x


# ── Sensor / measurement data ──────────────────────────────────────────────────

def get_sensor_data(
    n_sensors: int = N_SENSORS,
    noise_pct: float = NOISE_PCT,
    beam: str = "cantilever",
    seed: int = 42,
    device: str = "cpu",
):
    """
    Generate sparse, noisy deflection measurements.

    Returns
    -------
    x_sensors : torch.Tensor  (n_sensors, 1)  – sensor positions in [0, 1]
    w_noisy   : torch.Tensor  (n_sensors, 1)  – noisy deflection measurements
    w_clean   : torch.Tensor  (n_sensors, 1)  – noise-free reference
    """
    rng = np.random.default_rng(seed)
    x_np = np.sort(rng.uniform(0.0, 1.0, n_sensors))

    if beam == "cantilever":
        w_clean_np = analytical_w_cantilever(x_np)
    else:
        w_clean_np = analytical_w_simply_supported(x_np)

    # Gaussian noise scaled by noise_pct of the signal RMS
    rms = np.sqrt(np.mean(w_clean_np**2)) + 1e-12
    noise = rng.normal(0.0, noise_pct * rms, size=w_clean_np.shape)
    w_noisy_np = w_clean_np + noise

    x_t  = torch.tensor(x_np,      dtype=torch.float32, device=device).unsqueeze(1)
    w_n  = torch.tensor(w_noisy_np, dtype=torch.float32, device=device).unsqueeze(1)
    w_c  = torch.tensor(w_clean_np, dtype=torch.float32, device=device).unsqueeze(1)
    return x_t, w_n, w_c


# ── Boundary condition points ─────────────────────────────────────────────────

def get_boundary_points(device: str = "cpu"):
    """Return scalar tensors at x=0 and x=1 (normalised) with requires_grad=True."""
    x0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    x1 = torch.tensor([[1.0]], dtype=torch.float32, device=device, requires_grad=True)
    return x0, x1


# ── Dense evaluation grid (for plotting) ─────────────────────────────────────

def get_eval_grid(n: int = 1000, device: str = "cpu") -> torch.Tensor:
    """Dense evaluation grid in [0, 1], no grad required."""
    return torch.linspace(0.0, 1.0, n, device=device).unsqueeze(1)
