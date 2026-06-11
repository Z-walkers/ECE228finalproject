# Reproducing and Improving ST-PINN for PDE Solving

This repository contains a PyTorch reproduction and extension of ST-PINN for
one-dimensional PDE solving. The project compares vanilla PINN, ST-PINN, and
our improved variants on three PDE benchmarks:

- 1D Burgers equation
- 1D diffusion-reaction equation
- 1D diffusion-sorption equation

The original ST-PINN code was based on an old TensorFlow 1.15 environment. This
project ports the training pipeline to PyTorch and investigates whether
residual-guided pseudo-labeling remains robust under the reproduced setting.

## Project Structure

```text
source/
  Burgers1D_PINN.py          # Vanilla PINN for Burgers
  Burgers1D_STPINN.py        # ST-PINN / FFUSTPINN variants for Burgers
  DiffReact1D_PINN.py        # Vanilla PINN for DiffReact
  DiffReact1D_STPINN.py      # ST-PINN / FFUSTPINN variants for DiffReact
  DiffSorb1D_PINN.py         # Vanilla PINN for DiffSorb
  DiffSorb1D_STPINN.py       # ST-PINN / FFUSTPINN variants for DiffSorb
  run.py                     # Batch runner for experiments
  plot_results.py            # Plot generation script
  utilities.py               # Neural network, Fourier features, metrics, helpers
  pdes.py                    # PDE residual definitions

input/                       # PDE datasets
output/log/                  # Training logs
output/prediction/           # Saved prediction .npy files
output/figures*/             # Generated figures
backup_before_ffustpinn_*/   # Backup before FFUSTPINN modifications
```

## Methods

### PINN

The vanilla PINN is trained with initial condition loss, boundary condition loss,
PDE residual loss, and available data loss. The final prediction quality is
measured using relative L2 error:

```text
Relative L2 = ||u_pred - u_true||_2 / ||u_true||_2
```

### ST-PINN

ST-PINN adds residual-guided pseudo-labeling. It periodically ranks collocation
points by PDE residual and selects low-residual points as pseudo-label
candidates. These pseudo-labels are then added to the training objective as
extra supervision.

In our reproduction, ST-PINN does not consistently outperform vanilla PINN. This
suggests that low PDE residual alone is not always a reliable indicator of
prediction accuracy.

### USTPINN

USTPINN improves pseudo-label selection by adding an uncertainty-aware criterion.
In this implementation, uncertainty is estimated using teacher-student prediction
consistency. The teacher network is maintained as an exponential moving average
of the student network parameters, and the uncertainty score is defined as:

```text
U(x,t) = |u_student(x,t) - u_teacher(x,t)|
```

Pseudo-label candidates are accepted only when they have both low PDE residual
and low student-teacher disagreement.

### FFUSTPINN

FFUSTPINN further adds multi-frequency Fourier feature mapping to improve the
network's representation capacity. Instead of using only raw coordinates `(x,t)`,
the network input is augmented with sinusoidal Fourier features:

```text
gamma(x,t) = [x, t, sin(2*pi*f*x), cos(2*pi*f*x),
              sin(2*pi*f*t), cos(2*pi*f*t)]
```

This helps reduce spectral bias and improves performance on PDEs with more
complex spatiotemporal structures, such as Burgers and DiffSorb. However, it can
introduce unnecessary high-frequency components for smoother PDEs such as
DiffReact.

## Environment Setup

This project can be run with `uv`:

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

If the default PyTorch wheel does not match your GPU, reinstall PyTorch with the
CUDA index supported by your driver. For example:

```powershell
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch
```

Check the CUDA version supported by your driver with:

```powershell
nvidia-smi
```

## Running Experiments

Run all three PDE tasks with PINN, ST-PINN, and FFUSTPINN:

```powershell
.venv\Scripts\python.exe source\run.py --methods pinn stpinn ffustpinn
```

Run the 3 x 3 experiments concurrently:

```powershell
.venv\Scripts\python.exe source\run.py --methods pinn stpinn ffustpinn --parallel --workers 3
```

Run a short smoke test:

```powershell
.venv\Scripts\python.exe source\run.py --methods pinn stpinn ffustpinn --smoke --parallel --workers 3
```

Run a single task:

```powershell
.venv\Scripts\python.exe source\run.py --tasks burgers --methods pinn stpinn ffustpinn
```

Available tasks:

```text
burgers
diffreact
diffsorb
```

Available methods:

```text
pinn
stpinn
ffustpinn
dynamic-ffu
```

## Generating Figures

Generate comparison figures from the latest complete prediction files:

```powershell
.venv\Scripts\python.exe source\plot_results.py --methods pinn stpinn ffu-stpinn --out-dir output\figures_ffustpinn
```

Generate figures only for PINN and ST-PINN:

```powershell
.venv\Scripts\python.exe source\plot_results.py --methods pinn stpinn --out-dir output\figures_pinn_stpinn
```

The generated figures include:

- Burgers solution slices at selected time steps
- DiffReact training loss curves
- DiffSorb solution fields and point-wise error heatmaps

## Main Experimental Results

The latest complete FFUSTPINN run used timestamp `2026-06-05-08-01-41`.

### Relative L2 Error

| PDE | PINN | ST-PINN | USTPINN | FFUSTPINN |
|---|---:|---:|---:|---:|
| Burgers | 0.0844 | 0.0877 | 0.0831 | **0.0727** |
| DiffReact | **0.02667** | 0.02671 | 0.02673 | 0.03234 |
| DiffSorb | 0.0133 | 0.0136 | 0.0147 | **0.0110** |

## Observations

- ST-PINN does not consistently outperform vanilla PINN in the reproduced
  setting.
- USTPINN slightly improves Burgers, but the improvement is not consistent
  across all PDEs.
- FFUSTPINN clearly improves Burgers and DiffSorb, suggesting that Fourier
  features help with complex or structured spatiotemporal solutions.
- FFUSTPINN performs worse on DiffReact, likely because the solution is smoother
  and does not benefit from additional high-frequency coordinate features.

## Notes

- The final model used for prediction is restored from the best evaluation
  checkpoint rather than the last training checkpoint.
- Exact reproduction of the original ST-PINN paper may be difficult because the
  original datasets, preprocessing details, and TensorFlow 1.15 environment are
  not fully recoverable.
- Future work could add adaptive Fourier frequency filtering or uncertainty-based
  pseudo-label weighting to reduce unnecessary high-frequency artifacts.
