# Diffractive Cascades

Code accompanying the manuscript **"Diffractive cascades for polychromatic hard X-ray focusing"**.

This repository is organized as a reproducible pipeline with two main stages:

1. **Data generation (sweeps):** Run optimization sweeps that design diffractive cascades under various parameter configurations and save results to disk.
2. **Figure generation (notebooks):** Load the pre-computed optimization outputs and produce the manuscript figures.

Pre-generated data is included in `paper_data/`, so you can recreate all figures immediately using the notebooks without re-running any optimizations.

### Repository layout

- `src/` implements the forward models, optimization, and metrics.
- `paper/sweeps/` defines and runs the optimization sweeps that generate data.
- `paper/postprocess/` computes robustness metrics from sweep outputs.
- `paper/experiments/` runs one-off experiments (focal sweep comparison for depth of focus).
- `notebooks/` recreates manuscript figures from saved `.npz/.npy` data.
- `paper_data/` contains pre-generated optimization outputs consumed by the notebooks.
- `data/` contains tabulated optical constants (material refractive indices).
- `examples/` provides standalone scripts for quick single-optimization runs (useful for smoke tests or getting started).

### Naming convention

Notebooks and saved outputs are prefixed by the manuscript figure (and subfigure) they support, for example `fig2a_*` or `figA3b_*`. Sweep and experiment scripts write files of the form `{prefix}_params_{timestamp}.npy`, `{prefix}_sweep_arrays_{timestamp}.npy`, and `{prefix}_results_{timestamp}.npz` (or `{prefix}_results_{timestamp}_run_{i}.npz` for multi-run sweeps). The bundled files in `paper_data/` omit the timestamp suffix (e.g. `fig1_N_sweeps_params.npy`).

## Hardware Requirement

The optimization runs in this repo are computationally heavy. **Use a CUDA GPU** for practical runtimes.

- CPU-only runs are primarily useful for smoke tests.
- Sweeps and postprocessing scripts automatically use available GPUs.

## Installation

Use Python 3.11+ and install dependencies:

```bash
pip install -r requirements.txt
```

Set a data directory (recommended: `paper_data/`):

```bash
export DIFFRACTIVE_CASCADES_DATA_DIR=paper_data
```

## Recreating Figures from Existing Data

Open the notebook for the figure you want to reproduce and run all cells. Each notebook loads outputs from `paper_data/` using the matching `fig*` prefix.

| Figure | Notebook | Data prefix(es) in `paper_data/` |
|--------|----------|----------------------------------|
| Fig. 1(c) | `notebooks/fig1c_element_visualization.ipynb` | `fig1_N_sweeps` |
| Fig. 1(d) | `notebooks/fig1d_focal_spot_comparison.ipynb` | `fig1d_xray_focusing_testing` |
| Fig. 1(e) | `notebooks/fig1e_Nelem_sweep.ipynb` | `fig1_N_sweeps` |
| Fig. 2(a) | `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb` | `fig2a_bandwidth_energy_sweep`, `fig2a_thickness_energy_sweep` |
| Fig. 2(a) FZP cascade | `notebooks/fzp_cascade_energy_bandwidth_aspect_ratio.ipynb` | `fzp_cascade_bandwidth_energy_sweep`, `fzp_cascade_thickness_energy_sweep` |
| Fig. 2(b) | `notebooks/fig2b_mfs_sweep.ipynb` | `fig2b_Nelem_min_feature_sweep` |
| Fig. 2(c) | `notebooks/fig2c_aspect_ratio_scaling.ipynb` | `fig2c_thickness_energy_sweep` |
| Fig. 3(d) | `notebooks/fig3d_robustness.ipynb` | `fig3d_placement_robustness_results`, `fig3d_erosion_dilation_robustness_results`, `fig3d_thermal_robustness_results` |
| Fig. 4(b) | `notebooks/fig4b_depth_of_focus.ipynb` | `fig4b_xray_focusing_focal_sweep_comparison` |
| Fig. A.1 | `notebooks/figA1_partial_coherence.ipynb` | `figA1_coherence_illumination_sweep` |
| Fig. A.3(a) | `notebooks/figA3a_focal_length.ipynb` | `figA3a_focal_length_sweeps` |
| Fig. A.3(b) | `notebooks/figA3b_inter_elem_dist.ipynb` | `figA3b_inter_elem_dist_sweeps` |

Fig. 3(d) also uses sidewall data for Fig. A.4(b): `figA4b_sidewall_robustness_results`.

## Re-running Optimizations (Data Generation)

To regenerate the optimization data from scratch (requires a CUDA GPU for practical runtimes), use the paper entrypoint:

```bash
python paper/reproduce.py list
```

```bash
python paper/reproduce.py run <target_key>
```

You can override runtime settings:

```bash
python paper/reproduce.py run fig1e --data-dir paper_data --workers-per-gpu 2 --max-workers 8
```

### Main-text targets

- `fig1e` → `notebooks/fig1e_Nelem_sweep.ipynb`, data prefix `fig1_N_sweeps` (study `n_sweeps`).
- `fig1c` → `notebooks/fig1c_element_visualization.ipynb`, same `fig1_N_sweeps` run as `fig1e`.
- `fig2a_bandwidth` → `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb`, prefix `fig2a_bandwidth_energy_sweep` (study `bandwidth_energy`).
- `fig2a_thickness` → same notebook, prefix `fig2a_thickness_energy_sweep` (study `thickness_energy_fig2a`).
- `fzp_cascade_bandwidth` → `notebooks/fzp_cascade_energy_bandwidth_aspect_ratio.ipynb`, prefix `fzp_cascade_bandwidth_energy_sweep` (study `fzp_cascade_bandwidth_energy`).
- `fzp_cascade_thickness` → same notebook, prefix `fzp_cascade_thickness_energy_sweep` (study `fzp_cascade_thickness_energy`).
- `fig2b` → `notebooks/fig2b_mfs_sweep.ipynb`, prefix `fig2b_Nelem_min_feature_sweep` (study `nelem_min_feature`).
- `fig2c` → `notebooks/fig2c_aspect_ratio_scaling.ipynb`, prefix `fig2c_thickness_energy_sweep` (study `thickness_energy_main`).
- `fig3_placement`, `fig3_erosion_dilation`, `fig3_thermal` → `notebooks/fig3d_robustness.ipynb`; postprocess outputs `fig3d_*_robustness_results`.
- `figA4_sidewall` → same robustness notebook; postprocess output `figA4b_sidewall_robustness_results`.
- `fig4b` → `notebooks/fig4b_depth_of_focus.ipynb`, prefix `fig4b_xray_focusing_focal_sweep_comparison` (script `paper/experiments/xray_focusing_focal_sweep_comparison.py`).

`notebooks/fig1d_focal_spot_comparison.ipynb` (Fig. 1(d)) uses `fig1d_xray_focusing_testing_*` from `examples/xray_focusing_testing.py` and can be run from the included `paper_data/` without a separate `reproduce.py` target.

### Appendix targets

- `figA1` → `notebooks/figA1_partial_coherence.ipynb`, prefix `figA1_coherence_illumination_sweep` (study `coherence_illumination`).
- `figA3_focal` → `notebooks/figA3a_focal_length.ipynb`, prefix `figA3a_focal_length_sweeps` (study `focal_length`).
- `figA3_inter` → `notebooks/figA3b_inter_elem_dist.ipynb`, prefix `figA3b_inter_elem_dist_sweeps` (study `inter_element_distance`).

After re-running a sweep or experiment, update the timestamp `ID` in the corresponding notebook if it no longer matches the new `fig*_*_{timestamp}` filenames on disk.

## Examples

The `examples/` directory contains standalone scripts that run a single optimization:

```bash
python examples/xray_focusing_testing.py
python examples/xray_focusing_testing_partial_coherence.py
```

`xray_focusing_testing.py` writes `fig1d_xray_focusing_testing_params_{timestamp}.npy` and `fig1d_xray_focusing_testing_results_{timestamp}.npz` for `fig1d_focal_spot_comparison.ipynb`.

These scripts are useful for verifying your installation or experimenting with parameters outside of the full sweep pipeline. On a GPU, each completes in a few minutes. For batch jobs on a SLURM cluster, use `hpc/slurm/run_example_gpu.sh` (see [SLURM (GPU cluster jobs)](#slurm-gpu-cluster-jobs)).

### Console output

Runs print standardized, timestamped lines via `src/console.py`, for example:

```text
[14:32:01] [sweep] INFO: parallel pool: 2 GPU(s), 2 worker(s)/GPU, 4 worker(s) total
[14:32:01] [optimizer] INFO: beta stage 1/6: beta=1, constraints=off
```

Set `DIFFRACTIVE_CASCADES_QUIET=1` to suppress these messages.

## Direct Sweep Runner

If you want to run individual optimization sweeps directly (bypassing `reproduce.py`):

```bash
python paper/sweeps/run_sweep.py --list-studies
python paper/sweeps/run_sweep.py --study n_sweeps --save-dir paper_data
```

Study keys (`n_sweeps`, `bandwidth_energy`, `thickness_energy_fig2a`, etc.) map to the `fig*` save prefixes above. Each sweep runs a batch of optimizations over a parameter grid and saves the results with notebook-compatible axis naming and output schema. Projected element densities (`opt_rhos`) are stored as compressed `bool` arrays (binary solid/void); legacy `float64` files are still accepted by notebooks and postprocessing.

## Robustness Postprocessing Notes

Robustness scripts read a completed `fig1_N_sweeps` run and write figure-prefixed results:

| Script | Output prefix |
|--------|----------------|
| `paper/postprocess/placement_robustness.py` | `fig3d_placement_robustness_results` |
| `paper/postprocess/erosion_dilation_robustness.py` | `fig3d_erosion_dilation_robustness_results` |
| `paper/postprocess/thermal_robustness.py` | `fig3d_thermal_robustness_results` |
| `paper/postprocess/sidewall_robustness.py` | `figA4b_sidewall_robustness_results` |

By default, scripts read the bundled `paper_data/` files (`fig1_N_sweeps_params.npy`, `fig1_N_sweeps_sweep_arrays.npy`, `fig1_N_sweeps_results_run_0.npz`). After re-running the `n_sweeps` study, pass `--base-id <timestamp>` to use timestamped filenames instead:

```bash
python paper/postprocess/placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data
```

## SLURM (GPU cluster jobs)

The `hpc/slurm/` directory contains SLURM batch scripts for MIT-style clusters (Volta GPUs, Lmod modules). They are written to be submitted with `sbatch` from the **repository root**; you can also `cd hpc/slurm` and submit from there. Job stdout/stderr are written under `logs/slurm/` (created automatically; this directory is gitignored).

Before your first submission, open `hpc/slurm/_gpu_env.sh` and adjust the `module load` line (and any other site-specific setup) so the job sees Python 3.11+, PyTorch with CUDA, and this repo’s dependencies. The default module name is site-specific and will not exist on every cluster.

Edit the `#SBATCH` lines at the top of `run_sweep.sh`, `run_example_gpu.sh`, or `run_gpu_python.sh` if you need a different partition, GPU type, CPU count, or wall time.

### Files in `hpc/slurm/`

| File | Submit with `sbatch`? | Role |
|------|----------------------|------|
| `_gpu_env.sh` | No | Shared setup sourced by the launchers: `cd` to repo root, load Python/CUDA environment, create `logs/slurm/`. |
| `run_example_gpu.sh` | Yes | Single-GPU jobs for scripts in `examples/`. |
| `run_sweep.sh` | Yes | Multi-GPU jobs for full paper optimization sweeps via `paper/sweeps/run_sweep.py`. |
| `run_gpu_python.sh` | Yes | Generic GPU launcher for repository Python scripts (including `paper/postprocess/*.py`). |

SLURM copies submitted scripts into its spool directory, so the launchers **source** `_gpu_env.sh` by path under `SLURM_SUBMIT_DIR` rather than relying on `$0`. Do not run `_gpu_env.sh` as a standalone job.

### When to use which launcher

**`run_example_gpu.sh`** — short, single-optimization runs:

- Smoke-test the install after `pip install -r requirements.txt`.
- Regenerate Fig. 1(d) data (`examples/xray_focusing_testing.py` or `examples/xray_focusing_testing_partial_coherence.py`).
- Try parameter changes without launching a full parameter grid.

Default resources: **1** Volta GPU, **8** tasks (`#SBATCH --gres=gpu:volta:1`, `#SBATCH -n 8`). The script appends `--device cuda` unless you already pass `--device`.

```bash
# From repository root (recommended)
sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing.py
sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing_partial_coherence.py

# Optional args are forwarded to the Python script
sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing.py --device cuda:0
```

**`run_sweep.sh`** — manuscript-scale sweeps (many optimizations over a study grid):

- Regenerate sweep data for notebooks when bypassing `paper/reproduce.py` on the login node.
- Run one study at a time as a long GPU batch job (same studies as `python paper/reproduce.py run <target_key>`, but keyed by **study name**, not `fig*` target keys).

Default resources: **2** Volta GPUs, **40** tasks (`#SBATCH --gres=gpu:volta:2`, `#SBATCH -n 40`). The sweep runner defaults to **2 workers per GPU** (`run_sweep.py --workers-per-gpu 2`), so two GPUs typically run four parallel optimizations unless you override it.

List available study keys:

```bash
python paper/sweeps/run_sweep.py --list-studies
```

Submit a study (first argument is required; remaining arguments are passed through to `run_sweep.py`):

```bash
export DIFFRACTIVE_CASCADES_DATA_DIR=paper_data   # optional; or pass --save-dir

sbatch hpc/slurm/run_sweep.sh n_sweeps --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh bandwidth_energy --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh thickness_energy_fig2a --save-dir paper_data --workers-per-gpu 2 --max-workers 4
sbatch hpc/slurm/run_sweep.sh fzp_cascade_bandwidth_energy --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh fzp_cascade_thickness_energy --save-dir paper_data
```

Study keys align with the [re-running optimizations](#re-running-optimizations-data-generation) section, for example `n_sweeps` (Fig. 1e/c), `bandwidth_energy` and `thickness_energy_fig2a` (Fig. 2a), `nelem_min_feature` (Fig. 2b), `thickness_energy_main` (Fig. 2c), `fzp_cascade_bandwidth_energy` / `fzp_cascade_thickness_energy` (10-element FZP cascade vs optimized cascade), `coherence_illumination` (Fig. A.1), `focal_length` / `inter_element_distance` (Fig. A.3).

**`run_gpu_python.sh`** — generic launcher for robustness/postprocess and other repository Python scripts:

- Run robustness scripts on GPU with sweep-like `sbatch ... <script.py> [args...]` usage.
- Pass either a full repository-relative path or just a basename (the launcher resolves common locations like `paper/postprocess/`).

```bash
# Full path (explicit)
sbatch hpc/slurm/run_gpu_python.sh paper/postprocess/placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data

# Basename shorthand (resolved to paper/postprocess/placement_robustness.py)
sbatch hpc/slurm/run_gpu_python.sh placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data
```

`paper/reproduce.py` remains best for single-command orchestration, but `run_gpu_python.sh` is now the simplest direct `sbatch` path for standalone postprocess scripts.

### Monitoring jobs

```bash
squeue -u "$USER"
tail -f logs/slurm/<job-name>-<jobid>.out
```

After a sweep finishes, point the matching notebook at the new timestamp in the saved `fig*_*_{timestamp}` filenames (or set `DIFFRACTIVE_CASCADES_DATA_DIR` and update the notebook `ID` as described above).

## Citation

If you use this code, please cite:

William Michaels, Simo Pajovic, Joshua Chen, Charles Roques-Carmes, and Marin Soljačić, "Diffractive cascades for polychromatic hard X-ray focusing," arXiv:2605.15526, 2026.
