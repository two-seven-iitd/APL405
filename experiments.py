# experiments.py — Noise sensitivity + data sparsity studies
#
# Run from project root:
#   conda activate mixed-pinn
#   python experiments.py
#
# Outputs -> results/experiments/
#   noise_sensitivity.png
#   data_sparsity.png
#   terminal.log

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EPOCHS, NOISE_LEVELS, SENSOR_COUNTS, N_SENSORS, NOISE_PCT
from train import train_data_only, train_physics_only, train_hybrid
from evaluate import predict_w, compute_metrics
from data import get_eval_grid, analytical_w_cantilever


# ── Constants ────────────────────────────────────────────────────────────────
BEAM        = "cantilever"   # Primary beam per CONTEXT.md
OUT_DIR     = os.path.join("results", "experiments")
LOG_PATH    = os.path.join(OUT_DIR, "terminal.log")
SEEDS       = [42, 43, 44]   # 3 seeds -> averaged metrics (robustness check)


# ── Tee (duplicate stdout to log file) ───────────────────────────────────────
class _Tee:
    def __init__(self, path):
        self._file   = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, data):
        try:
            self._stdout.write(data)
        except UnicodeEncodeError:
            self._stdout.write(data.encode("ascii", errors="replace").decode("ascii"))
        self._file.write(data)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()
        sys.stdout = self._stdout


# ── Helpers ───────────────────────────────────────────────────────────────────
def _analytical_w(device="cpu"):
    x_np = np.linspace(0.0, 1.0, 1000)
    return analytical_w_cantilever(x_np)


def _rel_l2(net, w_true, device):
    w_pred = predict_w(net, device=device)
    return compute_metrics(w_pred, w_true)["rel_l2"]


def _train_and_score(mode, beam, noise_pct, n_sensors, device, epochs):
    """Train one model variant and return Rel L2 error (averaged over SEEDS)."""
    w_true = _analytical_w(device)
    scores = []
    for seed in SEEDS:
        if mode == "data_only":
            net, _ = train_data_only(
                device=device, epochs=epochs, beam=beam,
                n_sensors=n_sensors, noise_pct=noise_pct, seed=seed,
            )
            scores.append(_rel_l2(net, w_true, device))
        elif mode == "hybrid":
            w_net, _, _ = train_hybrid(
                device=device, epochs=epochs, beam=beam,
                n_sensors=n_sensors, noise_pct=noise_pct, seed=seed,
            )
            scores.append(_rel_l2(w_net, w_true, device))
    return float(np.mean(scores))


def _train_physics_once(beam, device, epochs):
    """Physics-Only doesn't depend on sensors/noise — train once as baseline."""
    w_net, _, _ = train_physics_only(device=device, epochs=epochs, beam=beam)
    w_true = _analytical_w(device)
    return _rel_l2(w_net, w_true, device)


# ── Experiment 1: Noise sensitivity ──────────────────────────────────────────
def run_noise_sensitivity(device, epochs):
    print("\n" + "=" * 60)
    print("  Experiment 1: Noise Sensitivity")
    print(f"  Beam: {BEAM}  |  N_sensors fixed: {N_SENSORS}")
    print(f"  Noise levels: {[f'{n*100:.0f}%' for n in NOISE_LEVELS]}")
    print("=" * 60)

    print("\n[Noise] Training Physics-Only baseline (once) ...")
    phys_score = _train_physics_once(BEAM, device, epochs)
    print(f"[Noise] Physics-Only Rel L2 = {phys_score*100:.2f}%  (noise-independent)")

    data_scores   = []
    hybrid_scores = []

    for noise in NOISE_LEVELS:
        pct_str = f"{noise*100:.0f}%"
        print(f"\n[Noise] --- noise={pct_str} ---")

        print(f"[Noise]   Training Data-Only  (noise={pct_str}) ...")
        s = _train_and_score("data_only", BEAM, noise, N_SENSORS, device, epochs)
        data_scores.append(s)
        print(f"[Noise]   Data-Only  Rel L2 = {s*100:.2f}%")

        print(f"[Noise]   Training Hybrid     (noise={pct_str}) ...")
        s = _train_and_score("hybrid", BEAM, noise, N_SENSORS, device, epochs)
        hybrid_scores.append(s)
        print(f"[Noise]   Hybrid     Rel L2 = {s*100:.2f}%")

    # Print summary table
    print("\n[Noise] Summary:")
    print(f"  {'Noise':>8}  {'Data-Only':>12}  {'Hybrid':>12}  {'Physics-Only':>14}")
    print("  " + "-" * 52)
    for i, noise in enumerate(NOISE_LEVELS):
        print(
            f"  {noise*100:>7.0f}%  "
            f"{data_scores[i]*100:>11.2f}%  "
            f"{hybrid_scores[i]*100:>11.2f}%  "
            f"{phys_score*100:>13.2f}%"
        )

    return NOISE_LEVELS, data_scores, hybrid_scores, phys_score


# ── Experiment 2: Data sparsity ───────────────────────────────────────────────
def run_data_sparsity(device, epochs):
    print("\n" + "=" * 60)
    print("  Experiment 2: Data Sparsity (Robustness to Limited Data)")
    print(f"  Beam: {BEAM}  |  Noise fixed: {NOISE_PCT*100:.0f}%")
    print(f"  Sensor counts: {SENSOR_COUNTS}")
    print("=" * 60)

    print("\n[Sparsity] Training Physics-Only baseline (once) ...")
    phys_score = _train_physics_once(BEAM, device, epochs)
    print(f"[Sparsity] Physics-Only Rel L2 = {phys_score*100:.2f}%  (sensor-independent)")

    data_scores   = []
    hybrid_scores = []

    for n_s in SENSOR_COUNTS:
        print(f"\n[Sparsity] --- n_sensors={n_s} ---")

        print(f"[Sparsity]   Training Data-Only  (n_sensors={n_s}) ...")
        s = _train_and_score("data_only", BEAM, NOISE_PCT, n_s, device, epochs)
        data_scores.append(s)
        print(f"[Sparsity]   Data-Only  Rel L2 = {s*100:.2f}%")

        print(f"[Sparsity]   Training Hybrid     (n_sensors={n_s}) ...")
        s = _train_and_score("hybrid", BEAM, NOISE_PCT, n_s, device, epochs)
        hybrid_scores.append(s)
        print(f"[Sparsity]   Hybrid     Rel L2 = {s*100:.2f}%")

    # Print summary table
    print("\n[Sparsity] Summary:")
    print(f"  {'Sensors':>8}  {'Data-Only':>12}  {'Hybrid':>12}  {'Physics-Only':>14}")
    print("  " + "-" * 52)
    for i, n_s in enumerate(SENSOR_COUNTS):
        print(
            f"  {n_s:>8}  "
            f"{data_scores[i]*100:>11.2f}%  "
            f"{hybrid_scores[i]*100:>11.2f}%  "
            f"{phys_score*100:>13.2f}%"
        )

    return SENSOR_COUNTS, data_scores, hybrid_scores, phys_score


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_noise_sensitivity(noise_levels, data_scores, hybrid_scores, phys_score, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [n * 100 for n in noise_levels]

    ax.plot(x, [s * 100 for s in data_scores],   "r-o", label="Data-Only",    linewidth=2, markersize=7)
    ax.plot(x, [s * 100 for s in hybrid_scores],  "g-o", label="Hybrid",       linewidth=2, markersize=7)
    ax.axhline(phys_score * 100, color="b", linestyle="--", linewidth=2, label="Physics-Only (baseline)")

    ax.set_xlabel("Noise level (%)", fontsize=12)
    ax.set_ylabel("Relative L² Error (%)", fontsize=12)
    ax.set_title(f"Noise Sensitivity — {BEAM.replace('_', ' ').title()} Beam", fontsize=13)
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "noise_sensitivity.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_data_sparsity(sensor_counts, data_scores, hybrid_scores, phys_score, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sensor_counts, [s * 100 for s in data_scores],   "r-o", label="Data-Only",    linewidth=2, markersize=7)
    ax.plot(sensor_counts, [s * 100 for s in hybrid_scores],  "g-o", label="Hybrid",       linewidth=2, markersize=7)
    ax.axhline(phys_score * 100, color="b", linestyle="--", linewidth=2, label="Physics-Only (baseline)")

    ax.set_xlabel("Number of sensors", fontsize=12)
    ax.set_ylabel("Relative L² Error (%)", fontsize=12)
    ax.set_title(f"Data Sparsity — {BEAM.replace('_', ' ').title()} Beam", fontsize=13)
    ax.set_xticks(sensor_counts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "data_sparsity.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tee = _Tee(LOG_PATH)
    sys.stdout = tee

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Epochs per run: {EPOCHS}  |  Seeds per config: {len(SEEDS)}")
    print(f"Estimated runs: {len(NOISE_LEVELS)*2 + 1 + len(SENSOR_COUNTS)*2 + 1} training jobs")

    # Experiment 1
    noise_levels, noise_data, noise_hybrid, noise_phys = run_noise_sensitivity(device, EPOCHS)
    plot_noise_sensitivity(noise_levels, noise_data, noise_hybrid, noise_phys, OUT_DIR)

    # Experiment 2
    sensor_counts, sparse_data, sparse_hybrid, sparse_phys = run_data_sparsity(device, EPOCHS)
    plot_data_sparsity(sensor_counts, sparse_data, sparse_hybrid, sparse_phys, OUT_DIR)

    print("\nAll experiments done.")
    tee.close()


if __name__ == "__main__":
    main()
