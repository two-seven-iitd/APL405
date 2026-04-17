# CONTEXT: Mixed-PINN Project

> **Read this entire file before writing any code.** It contains the complete specification for the project.

---

## 1. Project Summary

This is a course project for "Machine Learning in Mechanics" at IIT Delhi. It builds a **Mixed-Variable Physics-Informed Neural Network (Mixed-PINN)** that predicts beam deflection `w(x)` and internal bending moment `M(x)` for structural beams.

### Core Innovation
Standard PINNs compute 4th-order derivatives via autodiff → numerically unstable. This project reformulates into **two coupled 2nd-order equations** using M(x) as an auxiliary variable.

### The Data Story (Three-Way Comparison)
1. **Data-Only Model**: Standard NN trained only on 15 sparse noisy sensor measurements. No physics. Expected: poor.
2. **Physics-Only Mixed-PINN**: Trained on governing equations + BCs only. No data. Expected: matches analytical.
3. **Hybrid Mixed-PINN**: Trained on physics + noisy data. Expected: best overall, robust to noise.

---

## 2. Governing Equations

### Original (DO NOT implement this directly)
```
EI · d⁴w/dx⁴ = q(x)
```

### Mixed Reformulation (IMPLEMENT THIS)
```
Equation 1 (Equilibrium):    M''(x) = -q(x)
Equation 2 (Constitutive):   M(x) = -EI · w''(x)
```

### Boundary Conditions

**Cantilever Beam (Primary):**
- w(0) = 0, w'(0) = 0, M(L) = 0, M'(L) = 0

**Simply Supported Beam (Secondary):**
- w(0) = 0, w(L) = 0, M(0) = 0, M(L) = 0

### Analytical Solutions (for validation + synthetic data)

**Cantilever under uniform load q:**
```python
w(x) = -(q * x**2) / (24 * E * I) * (6*L**2 - 4*L*x + x**2)
M(x) = -(q * L**2) / 2 + q * L * x - (q * x**2) / 2
V(x) = q * L - q * x  # shear force, consistency check
```

**Simply Supported under uniform load q:**
```python
w(x) = -(q * x) / (24 * E * I) * (L**3 - 2*L*x**2 + x**3)
M(x) = (q * x) / 2 * (L - x)
```

### Sign Convention
- Downward load: positive q
- Downward deflection: negative w
- Sagging moment: positive M
- **BE CONSISTENT THROUGHOUT ALL CODE**

---

## 3. Parameters

### Beam Properties
| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Beam Length | L | 1.0 | m |
| Young's Modulus | E | 210e9 | Pa |
| Moment of Inertia | I | 8.33e-6 | m⁴ |
| Flexural Rigidity | EI | 1749.3 | N·m² |
| Distributed Load | q | 1000 | N/m |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Collocation Points | 500 uniformly in [0, 1] |
| Sensor Points | 15 random in [0, 1] |
| Noise Level | 10% Gaussian (also test 5%, 20%) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| LR Scheduler | ReduceLROnPlateau, patience=1000 |
| Epochs | 10,000 (increase to 20,000 if needed) |
| λ_Ω (Equilibrium) | 1.0 |
| λ_Υ (Coupling) | 1.0 |
| λ_Γ (Boundary) | 10.0 (may need 100 or 1000) |
| λ_D (Data) | 1.0 |

---

## 4. Neural Network Architecture

### Two Parallel Networks
```
w-net: Input(1) → 8 Hidden Layers × 64 neurons → Output(1)  [predicts w(x)]
M-net: Input(1) → 5 Hidden Layers × 64 neurons → Output(1)  [predicts M(x)]
```

- **Activation**: SiLU (preferred) or tanh — MUST be twice differentiable
- **Weight Init**: Xavier Uniform
- **Training**: Joint — single Adam optimizer over both networks' parameters
- **Input**: x normalised to [0, 1]

### CRITICAL: Automatic Differentiation
```python
# x MUST have requires_grad=True

# First derivative
w_x = torch.autograd.grad(
    outputs=w_pred, inputs=x,
    grad_outputs=torch.ones_like(w_pred),
    create_graph=True,     # MANDATORY for higher-order gradients
    retain_graph=True      # MANDATORY to keep graph for backprop
)[0]

# Second derivative
w_xx = torch.autograd.grad(
    outputs=w_x, inputs=x,
    grad_outputs=torch.ones_like(w_x),
    create_graph=True,
    retain_graph=True
)[0]
```

**Without `create_graph=True`, the model will NOT learn. This is the #1 PINN bug.**

---

## 5. Loss Function

```
L_Total = λ_Ω · L_Ω + λ_Υ · L_Υ + λ_Γ · L_Γ + λ_D · L_D
```

### L_Ω (Governing Equilibrium)
```python
L_omega = mean( (M_xx + q)**2 )  # over collocation points
```

### L_Υ (Coupling / Constitutive)
```python
L_upsilon = mean( (M_pred + EI * w_xx)**2 )  # over collocation points
```

### L_Γ (Boundary Conditions — Cantilever)
```python
L_gamma = w_net(0)**2 + w_x_at_0**2 + M_net(L)**2 + M_x_at_L**2
```

### L_D (Data Fidelity)
```python
L_data = mean( (w_net(x_sensors) - w_measured)**2 )
# w_measured = analytical_w(x_sensors) + gaussian_noise
```

---

## 6. Three Models to Train

| Model | Loss | Architecture |
|-------|------|-------------|
| A: Data-Only | L_D only | Single network (8×64, SiLU) |
| B: Physics-Only | L_Ω + L_Υ + L_Γ | Dual w-net + M-net |
| C: Hybrid | L_Ω + L_Υ + L_Γ + L_D | Dual w-net + M-net |

---

## 7. Required Outputs

### Plots (save as PNG, 300 DPI)
1. **Deflection Comparison**: Analytical (black dashed), Data-Only (red), Physics-Only (blue), Hybrid (green), sensors (red scatter)
2. **Moment Comparison**: Same layout for M(x)
3. **Pointwise Error**: |w_pred - w_analytical| for all three models
4. **Convergence**: Total loss vs epoch for all three
5. **Individual Losses**: L_Ω, L_Υ, L_Γ, L_D vs epoch for hybrid model

### Metrics (print table)
- MSE, Relative L² Error, Max Absolute Error — for all three models

### Optional Experiments
- Noise sensitivity: 5%, 10%, 20%
- Data sparsity: 5, 10, 15, 25 sensors
- Simply supported beam case

---

## 8. File Structure

```
mixed-pinn-project/
├── CONTEXT.md           # This file
├── config.py            # All parameters (beam + training + loss weights)
├── models.py            # WNet, MNet class definitions
├── losses.py            # All loss functions + derivative computation
├── data.py              # Collocation points, analytical solutions, sensor data
├── train.py             # Training loop for all three models
├── evaluate.py          # Metrics + comparison table
├── plot.py              # All visualisation
├── main.py              # Entry point
└── results/
    ├── plots/
    └── models/
```

### Libraries: torch, numpy, matplotlib. Nothing else.

---

## 9. Debugging Checklist

- [ ] All input tensors have `requires_grad=True`
- [ ] Autograd uses `create_graph=True` and `retain_graph=True`
- [ ] Both networks' params in same optimizer
- [ ] x normalised to [0, 1]
- [ ] Analytical solution plotted and verified before training
- [ ] All individual losses printed every 500 epochs
- [ ] λ_Γ high enough (10–1000) for BCs to be satisfied
- [ ] Sign conventions consistent everywhere
- [ ] Boundary points at x=0 and x=1 (normalised), not x=0 and x=L
