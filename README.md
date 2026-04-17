# Mixed-PINN Project — README

Course project for *Machine Learning in Mechanics*, IIT Delhi.
Predicts beam deflection `w(x)` and bending moment `M(x)` using a Mixed-Variable PINN.

---

## Quick Start

```bash
# Activate environment
conda activate mixed-pinn

# Train all three models (edit beam = "cantilever" or "simply_supported" in main.py line 87)
python main.py

# Generate derived quantity plots (shear, stress, strain) from saved weights
python derived_quantities/generate_plots.py --beam cantilever
```

Outputs go to `results/{beam}/` — plots, model weights, CSVs, and `terminal.log`.

---

## How to Work with Claude on This Project

Start every new conversation by giving Claude this context block so it picks up exactly where you left off:

```
Read CONTEXT.md and sessions/session_01_2026-03-29.md before responding.
The project is a Mixed-PINN for beam deflection (w) and moment (M).
Key decisions already made: EI=1749.3 hardcoded, coupling loss is M - EI*w'' (not M + EI*w''),
lambda_gamma=100 (cantilever) / 500 (simply_supported), 15k epochs, RESAMPLE_EVERY=50.
```

---

## Prompt Templates

### Starting a new session
```
Read CONTEXT.md and sessions/session_01_2026-03-29.md.
Today I want to [describe goal].
Don't re-explain what I already know — focus on what needs to change.
```

### Debugging a training run
```
Read CONTEXT.md and sessions/session_01_2026-03-29.md.
Here is the terminal.log output: [paste relevant lines]
Physics-Only Rel L2 is [X]%, Hybrid is [Y]%.
What is likely wrong and what should I change first?
```

### Adding a new experiment
```
Read CONTEXT.md.
I want to run the noise sensitivity experiment: 5%, 10%, 20% noise on the cantilever beam.
What is the minimal code change needed in config.py / main.py?
Show me only the diff, not the full file.
```

### Asking about a specific file
```
Read [filename].
Explain [function name] — focus on the physics/math motivation, not the syntax.
```

### Asking Claude to edit code
```
Read [filename].
Change [describe exactly what to change]. Keep everything else identical.
```

### After a training run — updating the session log
```
Read sessions/session_01_2026-03-29.md.
Add a new section for today (date: [YYYY-MM-DD]) covering:
- Beam: [cantilever / simply_supported]
- What I changed: [list]
- Results: Data-Only Rel L2=[X]%, Physics-Only=[Y]%, Hybrid=[Z]%
- Any new bugs found or fixed
```

---

## Tips for Best Results

- **Always cite the file and line** when asking Claude to change something — it prevents edits to the wrong location.
- **Paste terminal.log snippets** when debugging; Claude can read loss curves and spot convergence issues.
- **Say "show diff only"** to avoid Claude rewriting whole files unnecessarily.
- **Mention the beam type** (cantilever vs simply_supported) — loss weights differ between them.
- **Don't ask Claude to re-explain CONTEXT.md** — it already knows it. Ask about *decisions*, *tradeoffs*, or *what to change*.

---

## File Map

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Full project spec — read-only reference |
| `config.py` | All parameters: beam, training, loss weights, paths |
| `models.py` | WNet (8×64), MNet (5×64), DataOnlyNet (8×64) |
| `losses.py` | Autodiff helpers + all loss terms |
| `data.py` | Analytical solutions, sensor data, collocation points |
| `train.py` | Training loops for all 3 models |
| `evaluate.py` | MSE, Rel L2, Max Abs Error |
| `plot.py` | 5 required plots (300 DPI PNG) |
| `main.py` | Entry point — edit `beam` on line 87 to switch beam type |
| `derived_quantities/generate_plots.py` | Standalone: shear, stress, strain from saved weights |
| `tests/verify_signs.py` | Sign convention sanity check |
| `sessions/` | Session logs tracking all changes and results |
