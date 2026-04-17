# config.py — All parameters for the Mixed-PINN project
import os

# ── Beam properties ────────────────────────────────────────────────────────────
L  = 1.0       # Beam length [m]
E  = 210e9     # Young's modulus [Pa]
I  = 8.33e-6   # Second moment of area [m⁴]
EI = 1749.3    # Flexural rigidity [N·m²]  (from CONTEXT.md spec, NOT E*I)
q  = 1000.0    # Distributed load [N/m], positive = downward

# ── Collocation & sensor settings ─────────────────────────────────────────────
N_COLLOC  = 500   # Collocation points uniformly spaced in [0, 1]
N_SENSORS = 15    # Number of sparse sensor measurements
NOISE_PCT = 0.10  # Gaussian noise level (10 %)

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS        = 15_000
LR            = 1e-3
LR_PATIENCE   = 2000     # ReduceLROnPlateau patience
LR_FACTOR     = 0.5      # LR reduction factor
GRAD_CLIP     = 1.0      # Max gradient norm for clipping

# ── Collocation resampling ────────────────────────────────────────────────────
RESAMPLE_EVERY = 50   # Resample collocation points every N epochs (0 = fixed)

# ── Loss weights (beam-specific) ──────────────────────────────────────────────
# Cantilever: BCs are w(0)=0, w'(0)=0, M(L)=0, M'(L)=0
# Simply supported: BCs are w(0)=0, w(L)=0, M(0)=0, M(L)=0
#   - All 4 are Dirichlet (value) conditions, harder to enforce simultaneously
#   - Deflection ~10x smaller, so PDE residuals are smaller -> BC weight must be higher
LOSS_WEIGHTS = {
    "cantilever": {
        "lambda_omega":   1.0,
        "lambda_upsilon": 1.0,
        "lambda_gamma":   100.0,
        "lambda_data":    1.0,
    },
    "simply_supported": {
        "lambda_omega":   1.0,
        "lambda_upsilon": 1.0,
        "lambda_gamma":   500.0,
        "lambda_data":    1.0,
    },
}

def get_loss_weights(beam: str) -> dict:
    """Return loss weights for a given beam type."""
    return LOSS_WEIGHTS[beam]

# ── Architecture ──────────────────────────────────────────────────────────────
W_NET_HIDDEN_LAYERS = 8
M_NET_HIDDEN_LAYERS = 5
HIDDEN_NEURONS      = 64

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_EVERY = 500   # Print individual losses every N epochs

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR      = "results"

def get_beam_dirs(beam: str):
    """Return (plots_dir, models_dir, data_dir) for a given beam type."""
    base = os.path.join(RESULTS_DIR, beam)
    plots_dir  = os.path.join(base, "plots")
    models_dir = os.path.join(base, "models")
    data_dir   = os.path.join(base, "data")
    return plots_dir, models_dir, data_dir

# ── Noise sensitivity experiment values ───────────────────────────────────────
NOISE_LEVELS     = [0.05, 0.10, 0.20]
SENSOR_COUNTS    = [5, 10, 15, 25]
