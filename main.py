# main.py — Entry point for the Mixed-PINN project

import os
import sys
import numpy as np
import torch
from config import EPOCHS, N_SENSORS, NOISE_PCT, get_beam_dirs
from train import train_data_only, train_physics_only, train_hybrid
from evaluate import evaluate_all
from plot import save_all_plots
from data import (
    get_collocation_points,
    get_sensor_data,
    get_eval_grid,
    analytical_w_cantilever,
    analytical_M_cantilever,
    analytical_w_simply_supported,
    analytical_M_simply_supported,
)


class _Tee:
    """Duplicate stdout to a log file in real time."""
    def __init__(self, log_path):
        self._file = open(log_path, "w", encoding="utf-8")
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


def save_data_csvs(beam: str, data_dir: str, device: str = "cpu"):
    """Save analytical solutions, sensor data, and collocation points as CSVs."""
    os.makedirs(data_dir, exist_ok=True)

    # 1. Analytical solution on dense grid
    x_np = np.linspace(0.0, 1.0, 1000)
    if beam == "cantilever":
        w_ana = analytical_w_cantilever(x_np)
        M_ana = analytical_M_cantilever(x_np)
    else:
        w_ana = analytical_w_simply_supported(x_np)
        M_ana = analytical_M_simply_supported(x_np)

    ana = np.column_stack([x_np, w_ana, M_ana])
    np.savetxt(
        os.path.join(data_dir, "analytical_solution.csv"), ana,
        delimiter=",", header="x,w_analytical,M_analytical", comments="",
    )

    # 2. Sensor data (noisy measurements)
    x_s, w_noisy, w_clean = get_sensor_data(
        n_sensors=N_SENSORS, noise_pct=NOISE_PCT, beam=beam, device=device
    )
    sensor = np.column_stack([
        x_s.cpu().numpy().squeeze(),
        w_clean.cpu().numpy().squeeze(),
        w_noisy.cpu().numpy().squeeze(),
    ])
    np.savetxt(
        os.path.join(data_dir, "sensor_data.csv"), sensor,
        delimiter=",", header="x,w_clean,w_noisy", comments="",
    )

    # 3. Collocation points
    x_coll = get_collocation_points(device=device)
    coll = x_coll.detach().cpu().numpy().squeeze()
    np.savetxt(
        os.path.join(data_dir, "collocation_points.csv"), coll,
        delimiter=",", header="x", comments="",
    )

    print(f"Data CSVs saved to {data_dir}")


def main():
    beam = "simply_supported"   # change to "simply_supported" for secondary case

    plots_dir, models_dir, data_dir = get_beam_dirs(beam)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    tee = _Tee("terminal.log")
    sys.stdout = tee

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Beam type: {beam}\n")

    # ── Save data CSVs ──────────────────────────────────────────────────────
    save_data_csvs(beam, data_dir, device)

    # ── Train all three models ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Model A: Data-Only")
    print("=" * 60)
    data_net, hist_data = train_data_only(
        device=device, epochs=EPOCHS, beam=beam,
        n_sensors=N_SENSORS, noise_pct=NOISE_PCT,
    )

    print("\n" + "=" * 60)
    print("  Training Model B: Physics-Only Mixed-PINN")
    print("=" * 60)
    phys_w, phys_m, hist_phys = train_physics_only(
        device=device, epochs=EPOCHS, beam=beam,
    )

    print("\n" + "=" * 60)
    print("  Training Model C: Hybrid Mixed-PINN")
    print("=" * 60)
    hyb_w, hyb_m, hist_hyb = train_hybrid(
        device=device, epochs=EPOCHS, beam=beam,
        n_sensors=N_SENSORS, noise_pct=NOISE_PCT,
    )

    # ── Save model weights ───────────────────────────────────────────────────
    torch.save(data_net.state_dict(), os.path.join(models_dir, "data_only.pth"))
    torch.save(phys_w.state_dict(),   os.path.join(models_dir, "physics_w.pth"))
    torch.save(phys_m.state_dict(),   os.path.join(models_dir, "physics_m.pth"))
    torch.save(hyb_w.state_dict(),    os.path.join(models_dir, "hybrid_w.pth"))
    torch.save(hyb_m.state_dict(),    os.path.join(models_dir, "hybrid_m.pth"))
    print(f"\nModel weights saved to {models_dir}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Metrics")
    print("=" * 60)
    evaluate_all(data_net, phys_w, hyb_w, beam=beam, device=device)

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    save_all_plots(
        data_net, phys_w, phys_m, hyb_w, hyb_m,
        hist_data, hist_phys, hist_hyb,
        plots_dir, beam=beam, device=device,
    )
    print("Done.")
    tee.close()


if __name__ == "__main__":
    main()
