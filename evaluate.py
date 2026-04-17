# evaluate.py — Metrics and comparison table

import numpy as np
import torch
from data import get_eval_grid, analytical_w_cantilever, analytical_w_simply_supported


def predict_w(net, device: str = "cpu", n: int = 1000) -> np.ndarray:
    """Run a w-network or data-only network on a dense grid and return numpy array."""
    x_eval = get_eval_grid(n=n, device=device)
    with torch.no_grad():
        w_pred = net(x_eval).squeeze().cpu().numpy()
    return w_pred


def compute_metrics(w_pred: np.ndarray, w_true: np.ndarray) -> dict:
    """
    Compute three error metrics.

    Returns
    -------
    dict with keys: mse, rel_l2, max_abs
    """
    diff = w_pred - w_true
    mse     = float(np.mean(diff ** 2))
    rel_l2  = float(np.linalg.norm(diff) / (np.linalg.norm(w_true) + 1e-12))
    max_abs = float(np.max(np.abs(diff)))
    return {"mse": mse, "rel_l2": rel_l2, "max_abs": max_abs}


def print_metrics_table(results: dict, beam: str = "cantilever") -> None:
    """
    Print a comparison table.

    Parameters
    ----------
    results : dict  { model_name: {"mse": ..., "rel_l2": ..., "max_abs": ...} }
    """
    header = f"\n{'Model':<20}  {'MSE':>14}  {'Rel. L² Error':>16}  {'Max |Error|':>14}"
    print("=" * len(header))
    print(f"  Beam type: {beam.upper()}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<20}  {m['mse']:>14.4e}  {m['rel_l2']:>16.4e}  {m['max_abs']:>14.4e}"
        )
    print("=" * len(header))


def evaluate_all(
    data_only_net,
    physics_w_net,
    hybrid_w_net,
    beam: str = "cantilever",
    device: str = "cpu",
    n: int = 1000,
) -> dict:
    """Evaluate all three models and return a dict of metrics."""
    x_np = np.linspace(0.0, 1.0, n)

    if beam == "cantilever":
        w_true = analytical_w_cantilever(x_np)
    else:
        w_true = analytical_w_simply_supported(x_np)

    w_data    = predict_w(data_only_net,  device=device, n=n)
    w_physics = predict_w(physics_w_net,  device=device, n=n)
    w_hybrid  = predict_w(hybrid_w_net,   device=device, n=n)

    results = {
        "Data-Only":     compute_metrics(w_data,    w_true),
        "Physics-Only":  compute_metrics(w_physics, w_true),
        "Hybrid":        compute_metrics(w_hybrid,  w_true),
    }
    print_metrics_table(results, beam=beam)
    return results
