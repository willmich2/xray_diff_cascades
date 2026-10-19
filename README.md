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

Notebooks and saved outputs use the manuscript figure prefix, for example `fig2a_*` or `figS3b_*`. Paper generators default to the exact bundled names: `{prefix}_params.npy`, `{prefix}_sweep_arrays.npy`, and `{prefix}_results.npz` (or `{prefix}_results_run_{i}.npz` for studies with run suffixes). Re-running into the same directory replaces those files. Use `--data-dir` with `paper/reproduce.py` or `--save-dir` with `paper/sweeps/run_sweep.py` to keep a separate dataset directory.

For separately timestamped outputs, pass `--timestamp-outputs` to `paper/reproduce.py` or `paper/sweeps/run_sweep.py`. For standalone paper scripts, set `DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS=1`. Timestamped names insert `_{timestamp}` before the extension or `_run_{i}` suffix; update notebook input paths to read those files. Filename rules live in `paper/data_paths.py`.

## Hardware Requirement

The optimization runs in this repo are computationally heavy. **Use a CUDA GPU** for practical runtimes.

- CPU-only runs are primarily useful for smoke tests.
- Sweeps and postprocessing scripts automatically use available GPUs.

## Installation

Run commands from the repository root. Use Python 3.11+ and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Paper generators default to `paper_data/`. To set their data directory explicitly:

```bash
export DIFFRACTIVE_CASCADES_DATA_DIR=paper_data
```

## Recreating Figures from Existing Data

Open the notebook for the figure you want to reproduce in a Jupyter-compatible editor, select the Python environment with the installed dependencies, and run all cells. Start the kernel in the repository root or `notebooks/` so the notebooks can locate the source and data. Each notebook loads outputs from `paper_data/` using the matching `fig*` prefix. To use a different dataset directory, update the notebook's data paths; the notebooks do not read `DIFFRACTIVE_CASCADES_DATA_DIR`.

For JupyterLab, install and launch it from the repository root:

```bash
python -m pip install jupyterlab
python -m jupyterlab notebooks/
```

| Figure | Notebook | Data prefix(es) in `paper_data/` |
|--------|----------|----------------------------------|
| Fig. 1(c) | `notebooks/fig1c_element_visualization.ipynb` | `fig1_N_sweeps` |
| Fig. 1(d) | `notebooks/fig1d_focal_spot_comparison.ipynb` | `fig1_N_sweeps` |
| Fig. 1(e) | `notebooks/fig1e_Nelem_sweep.ipynb` | `fig1_N_sweeps` |
| Fig. 2(a) | `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb` | `fig2a_bandwidth_energy_sweep`, `fig2a_thickness_energy_sweep` |
| Fig. 2(b) | `notebooks/fig2b_mfs_sweep.ipynb` | `fig2b_Nelem_min_feature_sweep` |
| Fig. 2(c) | `notebooks/fig2c_aspect_ratio_scaling.ipynb` | `fig2a_thickness_energy_sweep` (shared with Fig. 2(a)) |
| Fig. 3(d) | `notebooks/fig3d_robustness.ipynb` | `fig3d_placement_robustness_results`, `fig3d_erosion_robustness_results`, `fig3d_thermal_robustness_results` |
| Fig. 4(b) | `notebooks/fig4b_depth_of_focus.ipynb` | `fig4b_xray_focusing_focal_sweep_comparison` |
| Fig. S1 | `notebooks/figS1_partial_coherence.ipynb` | `figS1_coherence_illumination_sweep` |
| Fig. S2 | `notebooks/figS2_physical_bound.ipynb` | `fig1_N_sweeps` |
| Fig. S3(a) | `notebooks/figS3a_focal_length.ipynb` | `figS3a_focal_length_sweeps` |
| Fig. S3(b) | `notebooks/figS3b_inter_elem_dist.ipynb` | `figS3b_inter_elem_dist_sweeps` |
| Fig. S4 | `notebooks/figS4_robustness.ipynb` | `figS4a_dilation_robustness_results`, `figS4b_sidewall_robustness_results`, `figS4c_tilt_robustness_results` |
| Fig. S5(a) | `notebooks/figS5a_fzp_cascade_nelem_sweep.ipynb` | `figS5a_fzp_cascade_nelem_sweep` |
| Fig. S5(b) | `notebooks/figS5b_fzp_cascade_energy_bandwidth_aspect_ratio.ipynb` | `figS5b_fzp_cascade_bandwidth_energy_sweep`, `figS5b_fzp_cascade_thickness_energy_sweep` |

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

Pass multiple target keys to run several figures, or use `all` to regenerate every target. Add `--dry-run` to preview the commands without running optimizations:

```bash
python paper/reproduce.py run all --dry-run
```

### Main-text targets

- `fig1e` → `notebooks/fig1e_Nelem_sweep.ipynb`, data prefix `fig1_N_sweeps` (study `n_sweeps`).
- `fig1c` → `notebooks/fig1c_element_visualization.ipynb`, same `fig1_N_sweeps` run as `fig1e`.
- `fig1d` → `notebooks/fig1d_focal_spot_comparison.ipynb`, same `fig1_N_sweeps` run as `fig1e`.
- `fig2a_bandwidth` → `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb`, prefix `fig2a_bandwidth_energy_sweep` (study `bandwidth_energy`).
- `fig2a_thickness` → same notebook, prefix `fig2a_thickness_energy_sweep` (study `thickness_energy`).
- `fig2b` → `notebooks/fig2b_mfs_sweep.ipynb`, prefix `fig2b_Nelem_min_feature_sweep` (study `nelem_min_feature`).
- `fig2c` → `notebooks/fig2c_aspect_ratio_scaling.ipynb`, shared prefix `fig2a_thickness_energy_sweep` (study `thickness_energy`).
- `fig3_placement`, `fig3_thermal` → `notebooks/fig3d_robustness.ipynb`; postprocess outputs `fig3d_placement_robustness_results` and `fig3d_thermal_robustness_results`.
- `fig3_erosion_dilation` → `notebooks/fig3d_robustness.ipynb` and `notebooks/figS4_robustness.ipynb`; postprocess outputs `fig3d_erosion_robustness_results` and `figS4a_dilation_robustness_results`, respectively.
- `fig4b` → `notebooks/fig4b_depth_of_focus.ipynb`, prefix `fig4b_xray_focusing_focal_sweep_comparison` (script `paper/experiments/xray_focusing_focal_sweep_comparison.py`).

Selecting several targets that share a generator (including `all`) runs that generator once.

### Supplementary targets

- `figS1` → `notebooks/figS1_partial_coherence.ipynb`, prefix `figS1_coherence_illumination_sweep` (study `coherence_illumination`).
- `figS2` → `notebooks/figS2_physical_bound.ipynb`, using `fig1_N_sweeps`.
- `figS3_focal` → `notebooks/figS3a_focal_length.ipynb`, prefix `figS3a_focal_length_sweeps` (study `focal_length`).
- `figS3_inter` → `notebooks/figS3b_inter_elem_dist.ipynb`, prefix `figS3b_inter_elem_dist_sweeps` (study `inter_element_distance`).
- `figS4_sidewall` → `notebooks/figS4_robustness.ipynb`, prefix `figS4b_sidewall_robustness_results`.
- `figS4_tilt` → the same notebook, prefix `figS4c_tilt_robustness_results`.
- `figS5a` → `notebooks/figS5a_fzp_cascade_nelem_sweep.ipynb`, prefix `figS5a_fzp_cascade_nelem_sweep` (script `paper/experiments/fzp_cascade_nelem_sweep.py`).
- `figS5b_bandwidth`, `figS5b_thickness` → `notebooks/figS5b_fzp_cascade_energy_bandwidth_aspect_ratio.ipynb`, prefixes `figS5b_fzp_cascade_bandwidth_energy_sweep` and `figS5b_fzp_cascade_thickness_energy_sweep`.

The old `figA1`, `figA3_focal`, `figA3_inter`, `figA4_sidewall`, `fzp_cascade_bandwidth`, and `fzp_cascade_thickness` target keys remain accepted as aliases.

Default outputs can be read by the notebooks without renaming. With timestamped output enabled, explicitly update notebook paths and pass `--base-n-sweep-id <timestamp>` for robustness processing of the chosen Fig. 1 run.

## Examples

The `examples/` directory contains standalone scripts that run a single optimization:

```bash
python examples/xray_focusing_testing.py
python examples/xray_focusing_testing_partial_coherence.py
```

`xray_focusing_testing.py` writes standalone `fig1d_xray_focusing_testing_params_{timestamp}.npy` and `fig1d_xray_focusing_testing_results_{timestamp}.npz` outputs. The current Fig. 1(d) notebook uses the shared `fig1_N_sweeps` dataset instead.

These scripts are useful for verifying your installation or experimenting with parameters outside of the full sweep pipeline. Runtime depends on the GPU and simulation settings; CPU runs can be much slower. For Slurm batch jobs, see the [optional submission templates](hpc/slurm/README.md).

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

Study keys (`n_sweeps`, `bandwidth_energy`, `thickness_energy`, etc.) map to the `fig*` save prefixes above. Each sweep runs a batch of optimizations over a parameter grid and saves the results with notebook-compatible axis naming and output schema. Projected element densities (`opt_rhos`) are stored as compressed `bool` arrays (binary solid/void); legacy `float64` files are still accepted by notebooks and postprocessing.

## Efficiency normalization

These are the current generators' efficiency denominators. Full pupil means `pi * (Nx / 2)**2` in pixel units. Active FZP area uses the fabrication-limited radius, clipped to the simulated pupil. Optimized cascades span the simulated pupil, so their denominator remains the full pupil even in configs specifying `active_device_area`.

| Generator | Optimized cascade | FZP baseline |
|-----------|-------------------|--------------|
| `bandwidth_energy_sweep.py`, `thickness_energy_sweep.py`, `nelem_min_feature_sweep.py`, `focal_length_sweeps.py` | Full pupil | Active area of the single FZP |
| `fzp_cascade_bandwidth_energy_sweep.py`, `fzp_cascade_thickness_energy_sweep.py` | Full pupil | Active entrance area of the FZP cascade (`radii_used[0]`) |
| `n_sweeps.py`, `coherence_illumination_sweep.py`, `inter_elem_dist_sweeps.py` | Full pupil | Full pupil |
| `fzp_cascade_comparison.py`, `fzp_cascade_nelem_sweep.py` | Full pupil | Full pupil for both the FZP cascade and single FZP |
| `xray_focusing_focal_sweep_comparison.py` | Full pupil | Full pupil |
| All five robustness scripts: placement, erosion/dilation, thermal, sidewall, tilt | Full pupil | Full pupil |
| Both `examples/xray_focusing_testing*.py` scripts | Full pupil | Full pupil |

The generic `sweep_framework.py` defaults to `full_pupil` and forwards config overrides to `src/util.py::compute_opt_and_fzp_metrics_2d`. That helper's `efficiency_normalization` option changes the automatically generated FZP baseline only. A primary design needs an explicit `design_normalization_radius_pixels` to change its denominator; `fzp_cascade_worker.py` supplies that radius for active-area FZP cascade sweeps.

## Robustness Postprocessing Notes

Robustness scripts read a completed `fig1_N_sweeps` run and write figure-prefixed results. Individual robustness targets do not generate their input data automatically. Use the bundled Fig. 1 data, or run `fig1e` first in the same data directory before processing a new dataset:

| Script | Output prefix |
|--------|----------------|
| `paper/postprocess/placement_robustness.py` | `fig3d_placement_robustness_results` |
| `paper/postprocess/erosion_dilation_robustness.py` | `fig3d_erosion_robustness_results`, `figS4a_dilation_robustness_results` |
| `paper/postprocess/thermal_robustness.py` | `fig3d_thermal_robustness_results` |
| `paper/postprocess/sidewall_robustness.py` | `figS4b_sidewall_robustness_results` |
| `paper/postprocess/tilt_robustness.py` | `figS4c_tilt_robustness_results` |

By default, scripts read the bundled `paper_data/` files (`fig1_N_sweeps_params.npy`, `fig1_N_sweeps_sweep_arrays.npy`, `fig1_N_sweeps_results_run_0.npz`). When the `n_sweeps` study was run with timestamped outputs, pass `--base-id <timestamp>` to select that run:

```bash
python paper/postprocess/placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data
```

## Optional: example Slurm submission scripts

The scripts in `hpc/slurm/` are templates for submitting GPU jobs on Slurm clusters. They include the author's cluster-specific environment module, Volta GPU requests, and resource allocations; adapt these settings to your own cluster before submitting jobs.

See [the Slurm guide](hpc/slurm/README.md) for setup and submission examples. On a GPU workstation or a cluster with another scheduler, use the Python commands above in your own environment or submission script. Recreating figures from the bundled data does not require these launchers.

## Tests

Run the filename, orchestration, and FZP comparison checks without launching optimization sweeps or requiring a GPU:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you use this code, please cite:

William Michaels, Simo Pajovic, Joshua Chen, Charles Roques-Carmes, and Marin Soljačić, "Diffractive cascades for polychromatic hard X-ray focusing," arXiv:2605.15526, 2026.
