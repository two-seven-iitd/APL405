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

from config import EPOCHS, NOISE_LEVELS, SENSOR_COUNTS, N_SENSORS, NOISE_PCT, get_beam_dirs
from train import train_data_only, train_hybrid
from evaluate import predict_w, compute_metrics
from data import analytical_w_cantilever
from models import WNet


# ── Constants ────────────────────────────────────────────────────────────────
BEAM        = "cantilever"   # Primary beam per CONTEXT.md
OUT_DIR     = os.path.join("results", "experiments")
LOG_PATH    = os.path.join(OUT_DIR, "terminal.log")
SEEDS       = [42, 43, 44, 45, 46]   # 5 seeds -> averaged metrics (reduces init variance)


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
def _analytical_w():
    """Analytical cantilever deflection on a 1000-point dense grid (numpy)."""
    x_np = np.linspace(0.0, 1.0, 1000)
    return analytical_w_cantilever(x_np)


def _rel_l2(net, w_true, device):
    w_pred = predict_w(net, device=device)
    return compute_metrics(w_pred, w_true)["rel_l2"]


def _seed_all(seed: int):
    """Seed every RNG that affects training reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # no-op on CPU, needed for GPU weight init
    np.random.seed(seed)


def _train_and_score(mode, beam, noise_pct, n_sensors, device, epochs):
    """Train one model variant across SEEDS and return (mean, std) Rel L2 error."""
    w_true = _analytical_w()
    scores = []
    for seed in SEEDS:
        _seed_all(seed)
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
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    return float(np.mean(scores)), float(np.std(scores))


def _load_physics_baseline(beam, device):
    """Load the already-trained Physics-Only w-net from results/{beam}/models/.

    Physics-Only doesn't depend on sensors or noise, so the baseline from
    main.py's run is reused here — avoids retraining and gives a stable,
    reproducible reference line (matches the 3.58% quoted in the session log).
    """
    _, models_dir, _ = get_beam_dirs(beam)
    weights_path = os.path.join(models_dir, "physics_w.pth")
    w_net = WNet().to(device)
    w_net.load_state_dict(torch.load(weights_path, map_location=device))
    w_net.eval()
    return _rel_l2(w_net, _analytical_w(), device)


# ── Experiment 1: Noise sensitivity ──────────────────────────────────────────
def run_noise_sensitivity(device, epochs):
    print("\n" + "=" * 60)
    print("  Experiment 1: Noise Sensitivity")
    print(f"  Beam: {BEAM}  |  N_sensors fixed: {N_SENSORS}")
    print(f"  Noise levels: {[f'{n*100:.0f}%' for n in NOISE_LEVELS]}")
    print("=" * 60)

    print("\n[Noise] Loading Physics-Only baseline from saved weights ...")
    phys_score = _load_physics_baseline(BEAM, device)
    print(f"[Noise] Physics-Only Rel L2 = {phys_score*100:.2f}%  (noise-independent, from saved weights)")

    data_scores,   data_stds   = [], []
    hybrid_scores, hybrid_stds = [], []

    for noise in NOISE_LEVELS:
        pct_str = f"{noise*100:.0f}%"
        print(f"\n[Noise] --- noise={pct_str} ---")

        print(f"[Noise]   Training Data-Only  (noise={pct_str}) x {len(SEEDS)} seeds ...")
        mean, std = _train_and_score("data_only", BEAM, noise, N_SENSORS, device, epochs)
        data_scores.append(mean); data_stds.append(std)
        print(f"[Noise]   Data-Only  Rel L2 = {mean*100:.2f}% ± {std*100:.2f}%")

        print(f"[Noise]   Training Hybrid     (noise={pct_str}) x {len(SEEDS)} seeds ...")
        mean, std = _train_and_score("hybrid", BEAM, noise, N_SENSORS, device, epochs)
        hybrid_scores.append(mean); hybrid_stds.append(std)
        print(f"[Noise]   Hybrid     Rel L2 = {mean*100:.2f}% ± {std*100:.2f}%")

    # Print summary table
    print("\n[Noise] Summary (mean ± std over seeds):")
    print(f"  {'Noise':>8}  {'Data-Only':>18}  {'Hybrid':>18}  {'Physics-Only':>14}")
    print("  " + "-" * 66)
    for i, noise in enumerate(NOISE_LEVELS):
        print(
            f"  {noise*100:>7.0f}%  "
            f"{data_scores[i]*100:>10.2f}% ± {data_stds[i]*100:>4.2f}%  "
            f"{hybrid_scores[i]*100:>10.2f}% ± {hybrid_stds[i]*100:>4.2f}%  "
            f"{phys_score*100:>13.2f}%"
        )

    return {
        "x_values":      NOISE_LEVELS,
        "data_means":    data_scores,
        "data_stds":     data_stds,
        "hybrid_means":  hybrid_scores,
        "hybrid_stds":   hybrid_stds,
        "physics_score": phys_score,
    }


# ── Experiment 2: Data sparsity ───────────────────────────────────────────────
def run_data_sparsity(device, epochs):
    print("\n" + "=" * 60)
    print("  Experiment 2: Data Sparsity (Robustness to Limited Data)")
    print(f"  Beam: {BEAM}  |  Noise fixed: {NOISE_PCT*100:.0f}%")
    print(f"  Sensor counts: {SENSOR_COUNTS}")
    print("=" * 60)

    print("\n[Sparsity] Loading Physics-Only baseline from saved weights ...")
    phys_score = _load_physics_baseline(BEAM, device)
    print(f"[Sparsity] Physics-Only Rel L2 = {phys_score*100:.2f}%  (sensor-independent, from saved weights)")

    data_scores,   data_stds   = [], []
    hybrid_scores, hybrid_stds = [], []

    for n_s in SENSOR_COUNTS:
        print(f"\n[Sparsity] --- n_sensors={n_s} ---")

        print(f"[Sparsity]   Training Data-Only  (n_sensors={n_s}) x {len(SEEDS)} seeds ...")
        mean, std = _train_and_score("data_only", BEAM, NOISE_PCT, n_s, device, epochs)
        data_scores.append(mean); data_stds.append(std)
        print(f"[Sparsity]   Data-Only  Rel L2 = {mean*100:.2f}% ± {std*100:.2f}%")

        print(f"[Sparsity]   Training Hybrid     (n_sensors={n_s}) x {len(SEEDS)} seeds ...")
        mean, std = _train_and_score("hybrid", BEAM, NOISE_PCT, n_s, device, epochs)
        hybrid_scores.append(mean); hybrid_stds.append(std)
        print(f"[Sparsity]   Hybrid     Rel L2 = {mean*100:.2f}% ± {std*100:.2f}%")

    # Print summary table
    print("\n[Sparsity] Summary (mean ± std over seeds):")
    print(f"  {'Sensors':>8}  {'Data-Only':>18}  {'Hybrid':>18}  {'Physics-Only':>14}")
    print("  " + "-" * 66)
    for i, n_s in enumerate(SENSOR_COUNTS):
        print(
            f"  {n_s:>8}  "
            f"{data_scores[i]*100:>10.2f}% ± {data_stds[i]*100:>4.2f}%  "
            f"{hybrid_scores[i]*100:>10.2f}% ± {hybrid_stds[i]*100:>4.2f}%  "
            f"{phys_score*100:>13.2f}%"
        )

    return {
        "x_values":      SENSOR_COUNTS,
        "data_means":    data_scores,
        "data_stds":     data_stds,
        "hybrid_means":  hybrid_scores,
        "hybrid_stds":   hybrid_stds,
        "physics_score": phys_score,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def _plot_experiment(results: dict, out_dir: str, *, xlabel: str, title: str,
                     fname: str, x_scale: float = 1.0):
    """Shared plotting logic for noise-sensitivity and data-sparsity figures."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [v * x_scale for v in results["x_values"]]

    ax.errorbar(x, [s*100 for s in results["data_means"]],
                yerr=[s*100 for s in results["data_stds"]],
                fmt="r-o", label="Data-Only", linewidth=2, markersize=7, capsize=4)
    ax.errorbar(x, [s*100 for s in results["hybrid_means"]],
                yerr=[s*100 for s in results["hybrid_stds"]],
                fmt="g-o", label="Hybrid", linewidth=2, markersize=7, capsize=4)
    ax.axhline(results["physics_score"] * 100, color="b", linestyle="--",
               linewidth=2, label="Physics-Only (baseline)")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Relative L² Error (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_noise_sensitivity(results, out_dir):
    _plot_experiment(
        results, out_dir,
        xlabel="Noise level (%)",
        title=f"Noise Sensitivity — {BEAM.replace('_', ' ').title()} Beam",
        fname="noise_sensitivity.png",
        x_scale=100.0,   # noise fractions -> percent
    )


def plot_data_sparsity(results, out_dir):
    _plot_experiment(
        results, out_dir,
        xlabel="Number of sensors",
        title=f"Data Sparsity — {BEAM.replace('_', ' ').title()} Beam",
        fname="data_sparsity.png",
        x_scale=1.0,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tee = _Tee(LOG_PATH)
    sys.stdout = tee

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Epochs per run: {EPOCHS}  |  Seeds per config: {len(SEEDS)}")
    n_runs = (len(NOISE_LEVELS) + len(SENSOR_COUNTS)) * 2 * len(SEEDS)
    print(f"Total training runs: {n_runs}  (Physics-Only loaded from saved weights — not retrained)")

    # Experiment 1
    noise_results = run_noise_sensitivity(device, EPOCHS)
    plot_noise_sensitivity(noise_results, OUT_DIR)

    # Experiment 2
    sparsity_results = run_data_sparsity(device, EPOCHS)
    plot_data_sparsity(sparsity_results, OUT_DIR)

    print("\nAll experiments done.")
    tee.close()


if __name__ == "__main__":
    main()
