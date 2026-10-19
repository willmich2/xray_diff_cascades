# Example Slurm submission scripts

These scripts show how GPU computations are submitted on the author's cluster. They provide reusable launch commands, but their environment setup and resource requests must be adapted to other Slurm clusters. For installation, figure reproduction, and commands that can run directly on a GPU workstation, see the [main README](../../README.md).

## Adapt the cluster settings

Before submitting jobs:

1. Edit `_gpu_env.sh` to load or activate your Python environment. The supplied `module load conda/Python-ML-2025b-pytorch` is specific to the author's cluster. Your environment needs Python 3.11+, PyTorch with CUDA, and the repository dependencies.
2. Edit the `#SBATCH` directives in the launcher you will use. Replace the `gpu:volta` request with your cluster's GPU request, choose suitable CPU and GPU allocations, and add any required account, partition, memory, or wall-time settings.
3. From the repository root, create the log directory before submission:

   ```bash
   mkdir -p logs/slurm
   ```

The launchers write stdout and stderr under `logs/slurm/`, which is gitignored. Creating it before submission makes it available when Slurm opens the job's output files.

The supplied resource requests are:

| Launcher | GPUs | Slurm tasks (`-n`) |
|----------|------|--------------------|
| `run_example_gpu.sh` | 1 Volta | 8 |
| `run_sweep.sh` | 2 Volta | 40 |
| `run_gpu_python.sh` | 2 Volta | 40 |

The Slurm task count does not configure the Python process pool. For sweeps, set parallelism with `--workers-per-gpu` and, optionally, `--max-workers`. The sweep runner defaults to two workers per GPU, so two visible GPUs normally run four optimizations in parallel. Match your CPU allocation to the workload and your cluster's scheduling guidance.

## Choose a launcher

Submit the examples below from the **repository root** after adapting the settings above. The launchers also support submission from `hpc/slurm/`; in that case, use the local launcher filename, such as `sbatch run_sweep.sh n_sweeps`.

### Single-optimization examples

Use `run_example_gpu.sh` to explore individual designs or check the installation:

```bash
sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing.py
sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing_partial_coherence.py
```

Edit simulation and optimization settings in the example Python scripts. The examples select CUDA automatically when available. Their outputs are standalone experiments; generate the shared Fig. 1 notebook inputs with the `n_sweeps` study.

### Optimization sweeps

Use `run_sweep.sh` for a parameter grid. Its first argument is a **study key**, such as `n_sweeps`, rather than a figure target such as `fig1e`. Remaining arguments are forwarded to `paper/sweeps/run_sweep.py`.

List the available studies without launching a sweep:

```bash
python paper/sweeps/run_sweep.py --list-studies
```

Submit a study:

```bash
sbatch hpc/slurm/run_sweep.sh n_sweeps --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh bandwidth_energy --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh thickness_energy --save-dir paper_data --workers-per-gpu 2 --max-workers 4
sbatch hpc/slurm/run_sweep.sh fzp_cascade_bandwidth_energy --save-dir paper_data
sbatch hpc/slurm/run_sweep.sh fzp_cascade_thickness_energy --save-dir paper_data
```

See the [reproduction targets](../../README.md#re-running-optimizations-data-generation) for the figure-to-study mapping. Use `--save-dir` or `DIFFRACTIVE_CASCADES_DATA_DIR` to select the output directory. Default filenames match the notebook inputs; if you use another directory or `--timestamp-outputs`, update the notebook input paths accordingly.

### Robustness and other Python scripts

Use `run_gpu_python.sh` to submit a repository Python script with its arguments:

```bash
sbatch hpc/slurm/run_gpu_python.sh paper/postprocess/placement_robustness.py --run-id 0 --data-dir paper_data
```

This example reads the bundled Fig. 1 data. For timestamped Fig. 1 inputs, add `--base-id <FIG1_N_SWEEP_ID>`. Robustness scripts require a completed `fig1_N_sweeps` dataset in the selected directory.

The launcher also accepts a basename such as `placement_robustness.py`, resolving it under common repository directories, including `paper/postprocess/`, `examples/`, `paper/experiments/`, and `paper/sweeps/`.

All three launchers source `_gpu_env.sh` to find the repository root and set up the environment. Submit them from the repository root or `hpc/slurm/` so they can locate that helper through `SLURM_SUBMIT_DIR`. The helper is sourced by the launchers; it is not a standalone batch job.

## Monitor jobs

```bash
squeue -u "$USER"
tail -f logs/slurm/<job-name>-<jobid>.out
```
