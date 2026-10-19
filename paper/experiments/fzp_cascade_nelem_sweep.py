"""Sweep optimized vs FZP-cascade vs single-FZP efficiency across element counts.

For each N, this is the same comparison as ``fzp_cascade_comparison.py`` at
aspect ratio 8:

- an optimized N-element cascade
- an N-element intermediate-field Fresnel zone-plate cascade with coinciding
  foci: the upstream plate fills the cascade aperture and downstream radii
  follow the first-order cone
- a single Fresnel zone plate at the last-element focal length (independent
  of N; evaluated once)

Each optimized cascade is repeated ``n_runs`` times (independent random
initializations); FZP designs are deterministic and evaluated once per N.

Submit on the cluster from the repository root:

    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_nelem_sweep.py --save-dir paper_data
    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_nelem_sweep.py --inter-elem-dist 1e-2
    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_nelem_sweep.py --opt-inter-elem-dist 1e-2 --fzp-inter-elem-dist 1e-3 --nelems 1 2 3 4 5 6 8 10 --n-runs 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import console
from src.util import get_formatted_datetime
from paper.experiments.fzp_cascade_common import (
    ASPECT_RATIO,
    FZPComparison,
    add_spacing_arguments,
    resolve_inter_elem_dists,
)
from paper.sweeps.density_io import pack_binary_density, save_sweep_results
from paper.data_paths import dataset_path, output_timestamp

_LOG = "fzp_cascade_nelem_sweep"
SAVE_PREFIX = "figS5a_fzp_cascade_nelem_sweep"
DEFAULT_NELEMS = (1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30)
DEFAULT_N_RUNS = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized cascade vs intermediate-field FZP cascade vs single FZP, "
            "swept over element count."
        )
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "paper_data"),
        help="Directory for params/results files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--nelems",
        type=int,
        nargs="+",
        default=list(DEFAULT_NELEMS),
        help=f"Element counts to sweep (default: {' '.join(map(str, DEFAULT_NELEMS))}).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=(
            "Independent optimization runs per element count "
            f"(default: {DEFAULT_N_RUNS})."
        ),
    )
    add_spacing_arguments(parser)
    return parser.parse_args()


def _save_outputs(
    save_dir: str,
    save_time: str,
    params_dict: dict,
    results_payload: dict,
) -> tuple[str, str]:
    timestamp = output_timestamp(save_time)
    params_path = dataset_path(save_dir, SAVE_PREFIX, "params", timestamp=timestamp)
    results_path = dataset_path(save_dir, SAVE_PREFIX, "results", timestamp=timestamp)
    np.save(params_path, params_dict)
    save_sweep_results(results_path, results_payload)
    return str(params_path), str(results_path)


def main() -> None:
    args = _parse_args()
    nelems = np.array(sorted(set(int(n) for n in args.nelems)), dtype=int)
    if nelems.size == 0 or np.any(nelems < 1):
        raise ValueError("--nelems must be one or more integers >= 1")
    n_runs = int(args.n_runs)
    if n_runs < 1:
        raise ValueError("--n-runs must be an integer >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    opt_dist, fzp_dist = resolve_inter_elem_dists(args)
    comparison = FZPComparison.from_defaults(device, opt_dist, fzp_dist)
    Nx = comparison.sim_params.Nx
    n_half = Nx // 2
    n_points = int(nelems.size)
    max_nelem = int(nelems.max())
    max_rho_len = max_nelem * n_half

    script_start_time = console.script_start(_LOG)
    for key, value in (("nelems", nelems.tolist()), ("n_runs", n_runs), ("Nx", Nx),
                       ("aspect_ratio", ASPECT_RATIO), ("opt_inter_elem_dist", opt_dist),
                       ("fzp_inter_elem_dist", fzp_dist), ("device", device)):
        console.kv(_LOG, key, value)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_time = get_formatted_datetime()

    console.banner(_LOG, "single FZP (independent of N)")
    fzp_x, fzp_metrics = comparison.single_fzp()
    fzp_gain = fzp_metrics["opt_gain"]
    fzp_efficiency = float(fzp_metrics["opt_efficiency"])
    fzp_width = float(fzp_metrics["opt_width"])
    fzp_obj = float(fzp_metrics["opt_final_obj"])
    fzp_intensity = np.asarray(fzp_metrics["opt_intensity_1d"])
    console.info(_LOG, f"fzp eff={fzp_efficiency:.6f} width={fzp_width} gain={fzp_gain:.4f}")

    opt_efficiencies = np.full((n_runs, n_points), np.nan, dtype=np.float64)
    fzp_cascade_efficiencies = np.full(n_points, np.nan, dtype=np.float64)
    opt_widths = np.full((n_runs, n_points), np.nan, dtype=np.float64)
    fzp_cascade_widths = np.full(n_points, np.nan, dtype=np.float64)
    opt_gains = np.full((n_runs, n_points), np.nan, dtype=np.float64)
    fzp_cascade_gains = np.full(n_points, np.nan, dtype=np.float64)
    opt_objs = np.full((n_runs, n_points), np.nan, dtype=np.float64)
    fzp_cascade_objs = np.full(n_points, np.nan, dtype=np.float64)
    opt_rhos = np.zeros((n_runs, n_points, max_rho_len), dtype=bool)
    fzp_cascade_rhos = np.zeros((n_points, max_rho_len), dtype=bool)
    opt_intensities = np.full((n_runs, n_points, Nx), np.nan, dtype=np.float64)
    fzp_cascade_intensities = np.full((n_points, Nx), np.nan, dtype=np.float64)
    fzp_focal_lengths = np.full((n_points, max_nelem), np.nan, dtype=np.float64)
    fzp_radii_theory = np.full((n_points, max_nelem), np.nan, dtype=np.float64)
    fzp_radii_used = np.full((n_points, max_nelem), np.nan, dtype=np.float64)

    params_dict = {**comparison.metadata(), "nelems": nelems, "n_runs": n_runs}

    def _payload() -> dict:
        return {
            "nelems": nelems,
            "n_runs": int(n_runs),
            "opt_rhos": pack_binary_density(opt_rhos),
            "fzp_cascade_rhos": pack_binary_density(fzp_cascade_rhos),
            "fzp_x": pack_binary_density(fzp_x.detach().cpu().numpy()),
            "opt_efficiencies": opt_efficiencies,
            "fzp_cascade_efficiencies": fzp_cascade_efficiencies,
            "fzp_efficiency": fzp_efficiency,
            "opt_widths": opt_widths,
            "fzp_cascade_widths": fzp_cascade_widths,
            "fzp_width": fzp_width,
            "opt_gains": opt_gains,
            "fzp_cascade_gains": fzp_cascade_gains,
            "fzp_gain": float(fzp_gain),
            "opt_objs": opt_objs,
            "fzp_cascade_objs": fzp_cascade_objs,
            "fzp_obj": fzp_obj,
            "opt_intensities": opt_intensities,
            "fzp_cascade_intensities": fzp_cascade_intensities,
            "fzp_intensity": fzp_intensity,
            "fzp_focal_lengths": fzp_focal_lengths,
            "fzp_radii_theory": fzp_radii_theory,
            "fzp_radii_used": fzp_radii_used,
        }

    for i, Nelem in enumerate(nelems.tolist()):
        console.banner(_LOG, f"N = {Nelem} ({i + 1}/{n_points})")
        opt_fwd_model_args = comparison.forward_args(Nelem)
        fzp_fwd_model_args = comparison.forward_args(Nelem, fzp=True)
        fzp_cascade_np, fzp_f, fzp_r_theory, fzp_r_used = comparison.cascade_design(Nelem)
        fzp_cascade_x = torch.as_tensor(fzp_cascade_np, dtype=torch.float64, device=device)
        fzp_cascade_packed = pack_binary_density(fzp_cascade_np).reshape(-1)
        fzp_cascade_rhos[i, : Nelem * n_half] = fzp_cascade_packed[: Nelem * n_half]
        fzp_focal_lengths[i, :Nelem] = fzp_f
        fzp_radii_theory[i, :Nelem] = fzp_r_theory
        fzp_radii_used[i, :Nelem] = fzp_r_used
        for j in range(Nelem):
            console.info(
                _LOG,
                (
                    f"N={Nelem} FZP[{j}] f={fzp_f[j]:.4e} m "
                    f"R_theory={fzp_r_theory[j]:.4e} m "
                    f"R_used={fzp_r_used[j]:.4e} m"
                ),
            )

        cascade_metrics = comparison.evaluate(fzp_cascade_x, fzp_fwd_model_args)
        fzp_cascade_gain = cascade_metrics["opt_gain"]
        fzp_cascade_efficiencies[i] = float(cascade_metrics["opt_efficiency"])
        fzp_cascade_widths[i] = float(cascade_metrics["opt_width"])
        fzp_cascade_gains[i] = float(fzp_cascade_gain)
        fzp_cascade_objs[i] = float(cascade_metrics["opt_final_obj"])
        cascade_I = np.asarray(cascade_metrics["opt_intensity_1d"]).reshape(-1)
        n_cascade_I = min(cascade_I.shape[0], fzp_cascade_intensities.shape[1])
        fzp_cascade_intensities[i, :n_cascade_I] = cascade_I[:n_cascade_I]
        console.info(
            _LOG,
            (
                f"N={Nelem} fzp_cascade eff={fzp_cascade_efficiencies[i]:.6f} "
                f"width={fzp_cascade_widths[i]} gain={fzp_cascade_gains[i]:.4f}"
            ),
        )
        if device.type == "cuda":
            del fzp_cascade_x
            torch.cuda.empty_cache()

        for run_id in range(n_runs):
            console.banner(
                _LOG, f"topology optimization N={Nelem} run {run_id + 1}/{n_runs}"
            )
            opt_start_time = time.time()
            rho_bar, _obj_list = comparison.optimize(Nelem)
            console.elapsed(
                _LOG, f"optimization N={Nelem} run {run_id + 1}/{n_runs}",
                time.time() - opt_start_time,
            )
            rho_bar_np = pack_binary_density(rho_bar.detach().cpu().numpy()).reshape(-1)
            opt_rhos[run_id, i, : Nelem * n_half] = rho_bar_np[: Nelem * n_half]

            console.banner(
                _LOG, f"post-optimization metrics N={Nelem} run {run_id + 1}/{n_runs}"
            )
            opt_metrics = comparison.evaluate(rho_bar, opt_fwd_model_args)
            opt_gain = opt_metrics["opt_gain"]
            opt_efficiencies[run_id, i] = float(opt_metrics["opt_efficiency"])
            opt_widths[run_id, i] = float(opt_metrics["opt_width"])
            opt_gains[run_id, i] = float(opt_gain)
            opt_objs[run_id, i] = float(opt_metrics["opt_final_obj"])
            opt_I = np.asarray(opt_metrics["opt_intensity_1d"]).reshape(-1)
            n_opt_I = min(opt_I.shape[0], opt_intensities.shape[-1])
            opt_intensities[run_id, i, :n_opt_I] = opt_I[:n_opt_I]

            console.info(
                _LOG,
                (
                    f"N={Nelem} run {run_id + 1}/{n_runs} "
                    f"opt eff={opt_efficiencies[run_id, i]:.6f} "
                    f"width={opt_widths[run_id, i]} gain={opt_gains[run_id, i]:.4f} | "
                    f"fzp_cascade eff={fzp_cascade_efficiencies[i]:.6f} "
                    f"width={fzp_cascade_widths[i]} gain={fzp_cascade_gains[i]:.4f} | "
                    f"fzp eff={fzp_efficiency:.6f} width={fzp_width} gain={fzp_gain:.4f}"
                ),
            )

            console.banner(
                _LOG, f"checkpoint save after N={Nelem} run {run_id + 1}/{n_runs}"
            )
            params_path, results_path = _save_outputs(
                save_dir, save_time, params_dict, _payload()
            )
            console.file_saved(_LOG, params_path)
            console.file_saved(_LOG, results_path)

            if device.type == "cuda":
                del rho_bar
                torch.cuda.empty_cache()

    console.banner(_LOG, "efficiency vs N")
    opt_eff_mean = np.nanmean(opt_efficiencies, axis=0)
    opt_eff_std = np.nanstd(opt_efficiencies, axis=0)
    for i, Nelem in enumerate(nelems.tolist()):
        console.info(
            _LOG,
            (
                f"N={Nelem:3d}  opt={opt_eff_mean[i]:.6f} ± {opt_eff_std[i]:.6f}  "
                f"fzp_cascade={fzp_cascade_efficiencies[i]:.6f}  "
                f"fzp={fzp_efficiency:.6f}"
            ),
        )

    console.script_done(_LOG, script_start_time)


if __name__ == "__main__":
    main()
