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
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import console
from src.forwardmodels import (
    forward_model_N_elements_mask,
    forward_model_N_elements_mask_2d_coherent_qdht,
)
from src.inversedesign_utils import fzp_cascade_half_profiles, zp_init
from src.optimizer import run_torch_optimization
from src.simparams import SimParams
from src.util import (
    compute_opt_and_fzp_metrics_2d,
    focusing_gain,
    gaussian_energy_spectrum,
    get_formatted_datetime,
)
from paper.sweeps.density_io import pack_binary_density, save_sweep_results
from paper.sweeps.standard_params import (
    BANDWIDTH_DEFAULT,
    BETA_SCHEDULE_DEFAULT,
    CENTRAL_ENERGY_EV_DEFAULT,
    CONSTRAINT_AGGREGATION_DEFAULT,
    CONSTRAINT_FAC_DEFAULT,
    CONSTRAINT_METHOD_DEFAULT,
    CROP_WIDTH_DEFAULT,
    DX_DEFAULT,
    EPSILON_DEFAULT,
    F_DEFAULT,
    FOCUSING_THRESHOLD_DEFAULT,
    GAP_MAP_DEFAULT,
    INTER_ELEM_DIST_DEFAULT,
    MATERIAL_DEFAULT,
    MATERIAL_MAP,
    MAX_EVAL_DEFAULT,
    MEMBRANE_MAP_SI3N4,
    MEMBRANE_THICKNESS_DEFAULT,
    MIN_BETA_DEFAULT,
    MIN_FEATURE_SIZE_DEFAULT,
    MORPH_AGG_BETA_DEFAULT,
    MORPH_BETA_DEFAULT,
    N_WVL_DEFAULT,
    NX_DEFAULT,
    P_DEFAULT,
    PARAM_TOLERANCE_DEFAULT,
    TOLERANCE_DEFAULT,
)

_LOG = "fzp_cascade_nelem_sweep"
SAVE_PREFIX = "fzp_cascade_nelem_sweep"
ASPECT_RATIO = 8.0
# Default FZP stacking gap for intermediate-field coinciding-foci plates
# (Gleber et al., Opt. Express 2014: 0.3–1 mm at 10 keV). The optimized
# cascade keeps INTER_ELEM_DIST_DEFAULT (1 cm) unless overridden.
FZP_INTER_ELEM_DIST_DEFAULT = 1e-3
DEFAULT_NELEMS = (1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30)
DEFAULT_N_RUNS = 3


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
        help="Element counts to sweep (default: 1 2 3 4 5 6 8 10).",
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
    parser.add_argument(
        "--inter-elem-dist",
        type=float,
        default=None,
        help=(
            "Inter-element distance (m) for both cascades. "
            "Overridden per cascade by --opt-inter-elem-dist / "
            "--fzp-inter-elem-dist."
        ),
    )
    parser.add_argument(
        "--opt-inter-elem-dist",
        type=float,
        default=None,
        help=(
            "Inter-element distance (m) for the optimized cascade "
            f"(default: {INTER_ELEM_DIST_DEFAULT:g})."
        ),
    )
    parser.add_argument(
        "--fzp-inter-elem-dist",
        type=float,
        default=None,
        help=(
            "Inter-element distance (m) for the FZP cascade "
            f"(default: {FZP_INTER_ELEM_DIST_DEFAULT:g})."
        ),
    )
    return parser.parse_args()


def _coinciding_focal_lengths(nelem: int, f: float, inter_elem_dist: float) -> list[float]:
    return [f + (nelem - 1 - i) * inter_elem_dist for i in range(nelem)]


def _resolve_inter_elem_dists(args: argparse.Namespace) -> tuple[float, float]:
    shared = args.inter_elem_dist
    opt_dist = INTER_ELEM_DIST_DEFAULT if args.opt_inter_elem_dist is None else args.opt_inter_elem_dist
    fzp_dist = (
        FZP_INTER_ELEM_DIST_DEFAULT
        if args.fzp_inter_elem_dist is None
        else args.fzp_inter_elem_dist
    )
    if shared is not None:
        if args.opt_inter_elem_dist is None:
            opt_dist = shared
        if args.fzp_inter_elem_dist is None:
            fzp_dist = shared
    opt_dist = float(opt_dist)
    fzp_dist = float(fzp_dist)
    if opt_dist <= 0.0 or fzp_dist <= 0.0:
        raise ValueError("inter-element distances must be > 0")
    return opt_dist, fzp_dist


def _save_outputs(
    save_dir: str,
    save_time: str,
    params_dict: dict,
    results_payload: dict,
) -> tuple[str, str]:
    params_ts = os.path.join(save_dir, f"{SAVE_PREFIX}_params_{save_time}.npy")
    results_ts = os.path.join(save_dir, f"{SAVE_PREFIX}_results_{save_time}.npz")
    params_stable = os.path.join(save_dir, f"{SAVE_PREFIX}_params.npy")
    results_stable = os.path.join(save_dir, f"{SAVE_PREFIX}_results.npz")

    np.save(params_ts, params_dict)
    save_sweep_results(results_ts, results_payload)
    shutil.copy2(params_ts, params_stable)
    shutil.copy2(results_ts, results_stable)
    return results_ts, results_stable


def main() -> None:
    args = _parse_args()
    nelems = np.array(sorted(set(int(n) for n in args.nelems)), dtype=int)
    if nelems.size == 0 or np.any(nelems < 1):
        raise ValueError("--nelems must be one or more integers >= 1")
    n_runs = int(args.n_runs)
    if n_runs < 1:
        raise ValueError("--n-runs must be an integer >= 1")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    material_map = MATERIAL_MAP
    gap_map = GAP_MAP_DEFAULT
    membrane_map = MEMBRANE_MAP_SI3N4

    N_wvl = N_WVL_DEFAULT
    central_energy_ev = CENTRAL_ENERGY_EV_DEFAULT
    bandwidth = BANDWIDTH_DEFAULT
    Nx = NX_DEFAULT
    dx = DX_DEFAULT
    f = F_DEFAULT
    opt_inter_elem_dist, fzp_inter_elem_dist = _resolve_inter_elem_dists(args)
    membrane_thickness = MEMBRANE_THICKNESS_DEFAULT
    min_feature_size = MIN_FEATURE_SIZE_DEFAULT
    element_thickness = ASPECT_RATIO * min_feature_size
    focusing_threshold = FOCUSING_THRESHOLD_DEFAULT
    crop_width = CROP_WIDTH_DEFAULT
    epsilon = EPSILON_DEFAULT
    tolerance = TOLERANCE_DEFAULT
    param_tolerance = PARAM_TOLERANCE_DEFAULT
    max_eval = MAX_EVAL_DEFAULT
    min_beta = MIN_BETA_DEFAULT
    constraint_fac = CONSTRAINT_FAC_DEFAULT
    P = P_DEFAULT
    constraint_method = CONSTRAINT_METHOD_DEFAULT
    constraint_aggregation = CONSTRAINT_AGGREGATION_DEFAULT
    morph_beta = MORPH_BETA_DEFAULT
    morph_agg_beta = MORPH_AGG_BETA_DEFAULT
    beta_schedule = list(BETA_SCHEDULE_DEFAULT)
    r_max = (Nx // 2) * dx
    n_half = Nx // 2
    center_offsets = None
    n_points = int(nelems.size)
    max_nelem = int(nelems.max())
    max_rho_len = max_nelem * n_half

    script_start_time = console.script_start(_LOG)
    console.kv(_LOG, "nelems", nelems.tolist())
    console.kv(_LOG, "n_runs", n_runs)
    console.kv(_LOG, "Nx", Nx)
    console.kv(_LOG, "aspect_ratio", ASPECT_RATIO)
    console.kv(_LOG, "element_thickness", element_thickness)
    console.kv(_LOG, "opt_inter_elem_dist", opt_inter_elem_dist)
    console.kv(_LOG, "fzp_inter_elem_dist", fzp_inter_elem_dist)
    console.kv(_LOG, "r_max", r_max)
    console.kv(_LOG, "device", device)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_time = get_formatted_datetime()
    console.info(_LOG, f"output directory {save_dir}")

    lams, weights = gaussian_energy_spectrum(
        central_energy_ev=central_energy_ev,
        N=N_wvl,
        bandwidth=bandwidth,
        device=device,
        bandwidth_in_wavelength=False,
    )
    sim_params = SimParams(
        Ny=1,
        Nx=Nx,
        dx=dx,
        device=device,
        dtype=torch.complex128,
        lams=lams,
        weights=weights,
    )
    elem_params = {
        "thickness": element_thickness,
        "elem_map": material_map,
        "gap_map": gap_map,
        "membrane_map": membrane_map,
        "membrane_thickness": membrane_thickness,
        "propagation_method": "angular",
    }
    opt_params_base = {
        "min_feature_size": min_feature_size / 2.0,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "param_tolerance": param_tolerance,
        "max_eval": max_eval,
        "min_beta": min_beta,
        "beta_schedule": beta_schedule,
        "constraint_fac": constraint_fac,
        "P": P,
        "constraint_method": constraint_method,
        "constraint_aggregation": constraint_aggregation,
        "morph_agg_beta": morph_agg_beta,
        "morph_beta": morph_beta,
    }

    n_center = int(2 * 1.22 * min_feature_size / dx)
    focusing_mask = torch.zeros(1, Nx, device=device)
    focusing_mask[0, Nx // 2 - n_center // 2 : Nx // 2 + n_center // 2] = 1.0
    lam_center = lams[lams.argmax()]

    metric_kwargs = dict(
        min_feature_size=min_feature_size,
        focusing_threshold=focusing_threshold,
        crop_width=crop_width,
        forward_model_1d=forward_model_N_elements_mask,
        forward_model_2d=forward_model_N_elements_mask_2d_coherent_qdht,
        zp_init_func=zp_init,
        compute_fzp=False,
    )

    console.banner(_LOG, "single FZP (independent of N)")
    fzp_profile = zp_init(lam_center, f, min_feature_size, 1, sim_params)
    fzp_x = (
        fzp_profile.to(device=device, dtype=torch.float64)
        if isinstance(fzp_profile, torch.Tensor)
        else torch.tensor(fzp_profile, dtype=torch.float64, device=device)
    )
    fzp_fwd_model_args = (
        elem_params,
        focusing_mask,
        torch.tensor([f], device=device, dtype=torch.float64),
        ((0.0, 0.0),),
    )
    fzp_metrics = compute_opt_and_fzp_metrics_2d(
        fzp_x, sim_params, fzp_fwd_model_args, **metric_kwargs
    )
    fzp_gain = focusing_gain(np.asarray(fzp_metrics["opt_intensity_1d"]), focusing_threshold)
    fzp_efficiency = float(fzp_metrics["opt_efficiency"])
    fzp_width = float(fzp_metrics["opt_width"])
    fzp_obj = float(fzp_metrics["opt_final_obj"])
    fzp_intensity = np.asarray(fzp_metrics["opt_intensity_1d"])
    console.info(
        _LOG,
        f"fzp eff={fzp_efficiency:.6f} width={fzp_width} gain={fzp_gain:.4f}",
    )

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

    params_dict = {
        "Nx": int(Nx),
        "dx": float(dx),
        "N_wvl": int(N_wvl),
        "central_energy_ev": float(central_energy_ev),
        "bandwidth": float(bandwidth),
        "min_feature_size": float(min_feature_size),
        "f": float(f),
        "inter_elem_dist": float(opt_inter_elem_dist),
        "opt_inter_elem_dist": float(opt_inter_elem_dist),
        "fzp_inter_elem_dist": float(fzp_inter_elem_dist),
        "membrane_thickness": float(membrane_thickness),
        "element_thickness": float(element_thickness),
        "aspect_ratio": float(ASPECT_RATIO),
        "r_max": float(r_max),
        "material": MATERIAL_DEFAULT,
        "nelems": nelems,
        "n_runs": int(n_runs),
        "focusing_threshold": float(focusing_threshold),
        "crop_width": int(crop_width),
        "optimizer": "run_torch_optimization",
        "metric_model_2d": "coherent_2d_qdht",
        "opt_params": {
            "epsilon": float(epsilon),
            "tolerance": float(tolerance),
            "param_tolerance": float(param_tolerance),
            "max_eval": int(max_eval),
            "min_beta": float(min_beta),
            "beta_schedule": list(beta_schedule),
            "constraint_fac": float(constraint_fac),
            "P": int(P),
            "constraint_method": constraint_method,
            "constraint_aggregation": constraint_aggregation,
            "morph_beta": float(morph_beta),
            "morph_agg_beta": float(morph_agg_beta),
        },
    }

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
        opt_params = dict(opt_params_base)
        opt_params["Nelem"] = int(Nelem)
        opt_z_dists = torch.tensor(
            (Nelem - 1) * (opt_inter_elem_dist,) + (f,),
            device=device,
            dtype=torch.float64,
        )
        fzp_z_dists = torch.tensor(
            (Nelem - 1) * (fzp_inter_elem_dist,) + (f,),
            device=device,
            dtype=torch.float64,
        )
        opt_fwd_model_args = (elem_params, focusing_mask, opt_z_dists, center_offsets)
        fzp_fwd_model_args = (elem_params, focusing_mask, fzp_z_dists, center_offsets)

        focal_lengths = _coinciding_focal_lengths(Nelem, f, fzp_inter_elem_dist)
        console.banner(
            _LOG, f"FZP cascade (intermediate-field, coinciding foci) N={Nelem}"
        )
        fzp_cascade_np, fzp_f, fzp_r_theory, fzp_r_used = fzp_cascade_half_profiles(
            lam_center,
            focal_lengths,
            min_feature_size,
            sim_params,
            max_radius=r_max,
        )
        fzp_cascade_x = torch.tensor(fzp_cascade_np, dtype=torch.float64, device=device)
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

        cascade_metrics = compute_opt_and_fzp_metrics_2d(
            fzp_cascade_x, sim_params, fzp_fwd_model_args, **metric_kwargs
        )
        fzp_cascade_gain = focusing_gain(
            np.asarray(cascade_metrics["opt_intensity_1d"]), focusing_threshold
        )
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
            raw_design, _obj_list, _intensity_list, _extra_list, model = (
                run_torch_optimization(
                    sim_params,
                    opt_params,
                    opt_fwd_model_args,
                    objective_function=forward_model_N_elements_mask,
                )
            )
            console.elapsed(
                _LOG,
                f"optimization N={Nelem} run {run_id + 1}/{n_runs}",
                time.time() - opt_start_time,
            )

            x_tensor = torch.tensor(raw_design, dtype=torch.float64, device=device)
            rho_tilde, _ = model.filter_density(x_tensor)
            rho_bar = (rho_tilde > 0.5).to(dtype=float)
            rho_bar_np = pack_binary_density(rho_bar.detach().cpu().numpy()).reshape(-1)
            opt_rhos[run_id, i, : Nelem * n_half] = rho_bar_np[: Nelem * n_half]

            console.banner(
                _LOG, f"post-optimization metrics N={Nelem} run {run_id + 1}/{n_runs}"
            )
            opt_metrics = compute_opt_and_fzp_metrics_2d(
                rho_bar, sim_params, opt_fwd_model_args, **metric_kwargs
            )
            opt_gain = focusing_gain(
                np.asarray(opt_metrics["opt_intensity_1d"]), focusing_threshold
            )
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
            results_ts, results_stable = _save_outputs(
                save_dir, save_time, params_dict, _payload()
            )
            console.file_saved(_LOG, results_ts)
            console.file_saved(_LOG, results_stable)

            if device.type == "cuda":
                del x_tensor, rho_tilde, rho_bar, model, raw_design
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
