"""Compare an optimized diffractive cascade to an intermediate-field FZP cascade.

Optimizes an N-element cascade (default N=5) at aspect ratio 8, then evaluates:
- the optimized cascade
- an N-element intermediate-field Fresnel zone-plate cascade with coinciding
  foci: the upstream plate fills the cascade aperture and downstream radii
  follow the first-order cone
- a single Fresnel zone plate at the last-element focal length (paper baseline)

Submit on the cluster from the repository root:

    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_comparison.py --save-dir paper_data
    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_comparison.py --inter-elem-dist 1e-2
    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/fzp_cascade_comparison.py --opt-inter-elem-dist 1e-2 --fzp-inter-elem-dist 1e-3
"""

from __future__ import annotations

import argparse
import os
import shutil
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

_LOG = "fzp_cascade_comparison"
SAVE_PREFIX = "fzp_cascade_comparison"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized cascade vs intermediate-field FZP cascade vs single FZP."
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
        "--nelem",
        type=int,
        default=5,
        help="Number of cascade elements (same for optimized and FZP stacks).",
    )
    add_spacing_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.nelem < 1:
        raise ValueError("--nelem must be >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    opt_dist, fzp_dist = resolve_inter_elem_dists(args)
    comparison = FZPComparison.from_defaults(device, opt_dist, fzp_dist)
    Nelem = int(args.nelem)
    opt_fwd_model_args = comparison.forward_args(Nelem)
    fzp_fwd_model_args = comparison.forward_args(Nelem, fzp=True)
    opt_z_dists = opt_fwd_model_args[2]
    fzp_z_dists = fzp_fwd_model_args[2]
    focusing_mask = comparison.focusing_mask

    script_start_time = console.script_start(_LOG)
    for key, value in (("Nelem", Nelem), ("Nx", comparison.sim_params.Nx),
                       ("aspect_ratio", ASPECT_RATIO), ("opt_inter_elem_dist", opt_dist),
                       ("fzp_inter_elem_dist", fzp_dist), ("device", device)):
        console.kv(_LOG, key, value)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_time = get_formatted_datetime()

    console.banner(_LOG, "topology optimization")
    opt_start_time = time.time()
    rho_bar, obj_list_np = comparison.optimize(Nelem)
    console.elapsed(_LOG, "optimization", time.time() - opt_start_time)

    console.banner(_LOG, "FZP cascade (intermediate-field, coinciding foci)")
    fzp_cascade_np, fzp_focal_lengths, fzp_radii_theory, fzp_radii_used = comparison.cascade_design(Nelem)
    fzp_cascade_x = torch.as_tensor(fzp_cascade_np, dtype=torch.float64, device=device)
    for i in range(Nelem):
        console.info(
            _LOG, f"FZP[{i}] f={fzp_focal_lengths[i]:.4e} m "
            f"R_theory={fzp_radii_theory[i]:.4e} m R_used={fzp_radii_used[i]:.4e} m",
        )

    console.banner(_LOG, "post-optimization metrics")
    opt_metrics = comparison.evaluate(rho_bar, opt_fwd_model_args, compute_fzp=True)
    cascade_metrics = comparison.evaluate(fzp_cascade_x, fzp_fwd_model_args)
    opt_gain = opt_metrics["opt_gain"]
    fzp_cascade_gain = cascade_metrics["opt_gain"]
    fzp_gain = opt_metrics["fzp_gain"]
    console.info(
        _LOG, f"opt eff={opt_metrics['opt_efficiency']:.6f} | "
        f"fzp_cascade eff={cascade_metrics['opt_efficiency']:.6f} | "
        f"fzp eff={opt_metrics['fzp_efficiency']:.6f}",
    )
    params_dict = {**comparison.metadata(), "Nelem": Nelem}

    results_payload = {
        "rho_bar": pack_binary_density(rho_bar.detach().cpu().numpy()),
        "fzp_cascade_rhos": pack_binary_density(fzp_cascade_np),
        "fzp_x": pack_binary_density(opt_metrics["fzp_x"].detach().cpu().numpy()),
        "opt_intensity": np.asarray(opt_metrics["opt_intensity_1d"]),
        "opt_efficiency": float(opt_metrics["opt_efficiency"]),
        "opt_obj": float(opt_metrics["opt_final_obj"]),
        "opt_width": float(opt_metrics["opt_width"]),
        "opt_gain": float(opt_gain),
        "fzp_cascade_intensity": np.asarray(cascade_metrics["opt_intensity_1d"]),
        "fzp_cascade_efficiency": float(cascade_metrics["opt_efficiency"]),
        "fzp_cascade_obj": float(cascade_metrics["opt_final_obj"]),
        "fzp_cascade_width": float(cascade_metrics["opt_width"]),
        "fzp_cascade_gain": float(fzp_cascade_gain),
        "fzp_intensity": np.asarray(opt_metrics["fzp_intensity_1d"]),
        "fzp_efficiency": float(opt_metrics["fzp_efficiency"]),
        "fzp_obj": float(opt_metrics["fzp_final_obj"]),
        "fzp_width": float(opt_metrics["fzp_width"]),
        "fzp_gain": float(fzp_gain),
        "fzp_focal_lengths": fzp_focal_lengths,
        "fzp_radii_theory": fzp_radii_theory,
        "fzp_radii_used": fzp_radii_used,
        "obj_list": obj_list_np,
        "z_dists": opt_z_dists.detach().cpu().numpy(),
        "opt_z_dists": opt_z_dists.detach().cpu().numpy(),
        "fzp_z_dists": fzp_z_dists.detach().cpu().numpy(),
        "mask": focusing_mask.detach().cpu().numpy(),
    }

    console.banner(_LOG, "saving results")
    params_ts = os.path.join(save_dir, f"{SAVE_PREFIX}_params_{save_time}.npy")
    results_ts = os.path.join(save_dir, f"{SAVE_PREFIX}_results_{save_time}.npz")
    params_stable = os.path.join(save_dir, f"{SAVE_PREFIX}_params.npy")
    results_stable = os.path.join(save_dir, f"{SAVE_PREFIX}_results.npz")

    np.save(params_ts, params_dict)
    save_sweep_results(results_ts, results_payload)
    shutil.copy2(params_ts, params_stable)
    shutil.copy2(results_ts, results_stable)

    console.file_saved(_LOG, results_ts)
    console.file_saved(_LOG, results_stable)
    console.script_done(_LOG, script_start_time)


if __name__ == "__main__":
    main()
