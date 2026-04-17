"""Quick sign-convention verification — no training needed.

Plugs the ANALYTICAL solutions into each PDE residual and checks
whether they are zero.  If any residual is large, the sign convention
is wrong in that term.
"""

import torch
import numpy as np
from config import EI, q, L

# Dense test points
x_np = np.linspace(0.0, 1.0, 200)
x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

# ── Analytical solutions (from data.py / CONTEXT.md) ─────────────────────────
# w(x) negative (downward), M(x) = -q(L-x)^2 / 2 (negative at root)
w_exact = -(q * x**2) / (24 * EI) * (6*L**2 - 4*L*x + x**2)
M_exact = -(q * L**2) / 2 + q * L * x - (q * x**2) / 2

# ── Autodiff derivatives ─────────────────────────────────────────────────────
def d(y, x_):
    return torch.autograd.grad(y, x_, torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

w_x  = d(w_exact, x)
w_xx = d(w_x, x)
M_x  = d(M_exact, x)
M_xx = d(M_x, x)

# ── Check each residual ─────────────────────────────────────────────────────

# 1. Equilibrium: M'' + q = 0
res_eq = (M_xx + q)
print("=== Residual checks (should all be ~0) ===\n")
print(f"Equilibrium  M'' + q          max|res| = {res_eq.abs().max().item():.2e}")

# 2. Coupling variant A:  M + EI*w'' = 0   (CONTEXT.md literal)
res_coupA = M_exact + EI * w_xx
print(f"Coupling(A)  M + EI·w''       max|res| = {res_coupA.abs().max().item():.2e}")

# 3. Coupling variant B:  M - EI*w'' = 0   (current code after fix)
res_coupB = M_exact - EI * w_xx
print(f"Coupling(B)  M - EI·w''       max|res| = {res_coupB.abs().max().item():.2e}")

# ── BCs ──────────────────────────────────────────────────────────────────────
print(f"\n=== Boundary conditions ===")
print(f"w(0)   = {w_exact[0].item():.6e}   (should be 0)")
print(f"w'(0)  = {w_x[0].item():.6e}   (should be 0)")
print(f"M(L)   = {M_exact[-1].item():.6e}   (should be 0)")
print(f"M'(L)  = {M_x[-1].item():.6e}   (should be 0)")

# ── Tip deflection sanity ────────────────────────────────────────────────────
w_tip = w_exact[-1].item()
w_tip_expected = -q * L**4 / (8 * EI)
print(f"\n=== Tip deflection ===")
print(f"w(L)          = {w_tip:.6e} m")
print(f"Expected      = {w_tip_expected:.6e} m")

print("\n>>> If Coupling(A) is ~0, use M + EI*w'' in loss")
print(">>> If Coupling(B) is ~0, use M - EI*w'' in loss")
