"""Single-plane vs prescribed-chromatic cascade focusing.

Optimizes two N-element cascades on the same spectrum:

1. The existing intensity-mask objective (all wavelengths scored at one plane).
2. ``forward_model_N_elements_mask_chromatic``, which sends each sampled
   energy to its own last-hop distance (linear in energy, FZP-like sign).

Then scans per-wavelength intensity vs defocus for both designs and a
single Fresnel zone plate, and records in-mask crosstalk at the prescribed
planes.

Submit from the repository root:

    sbatch hpc/slurm/run_gpu_python.sh paper/experiments/chromatic_focusing.py
    python paper/experiments/chromatic_focusing.py --save-dir paper_data
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
from src.forwardmodels import (
    forward_model_N_elements_mask,
    forward_model_N_elements_mask_chromatic,
    scan_cascade_intensity_z,
)
from src.inversedesign_utils import zp_init
from src.optimizer import run_torch_optimization
from src.simparams import SimParams
from src.util import (
    gaussian_energy_spectrum,
    get_formatted_datetime,
    linear_energy_focal_shift,
    wavelengths_to_energy_ev,
)
from paper.sweeps.density_io import pack_binary_density, save_sweep_results
from paper.sweeps.standard_params import (
    BETA_SCHEDULE_DEFAULT,
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
    N_ELEMENTS_DEFAULT,
    NX_DEFAULT,
    P_DEFAULT,
    PARAM_TOLERANCE_DEFAULT,
    TOLERANCE_DEFAULT,
    ELEMENT_THICKNESS_DEFAULT,
    CENTRAL_ENERGY_EV_DEFAULT,
)

_LOG = "chromatic_focusing"
SAVE_PREFIX = "chromatic_focusing"

# Wide enough that FZP chromatic walk exceeds the diffraction-limited DoF.
BANDWIDTH_CHROMATIC = 1e-2
N_WVL_CHROMATIC = 5
CHROMATIC_HALF_WIDTH_DEFAULT = 500e-6
N_Z_EVAL_DEFAULT = 81


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize and diagnose single-plane vs chromatic cascade focusing."
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "paper_data"),
        help="Directory for params/results files.",
    )
    parser.add_argument("--device", default=None, help="Torch device.")
    parser.add_argument("--nelem", type=int, default=N_ELEMENTS_DEFAULT)
    parser.add_argument("--n-wvl", type=int, default=N_WVL_CHROMATIC)
    parser.add_argument("--bandwidth", type=float, default=BANDWIDTH_CHROMATIC)
    parser.add_argument(
        "--chromatic-half-width",
        type=float,
        default=CHROMATIC_HALF_WIDTH_DEFAULT,
        help="Half-span (m) of the prescribed last-hop distances around f.",
    )
    parser.add_argument("--n-z-eval", type=int, default=N_Z_EVAL_DEFAULT)
    parser.add_argument("--max-eval", type=int, default=MAX_EVAL_DEFAULT)
    return parser.parse_args()


def _project_design(raw_design, model, device):
    x_tensor = torch.tensor(raw_design, dtype=torch.float64, device=device)
    rho_tilde, _ = model.filter_density(x_tensor)
    return (rho_tilde > 0.5).to(dtype=torch.float64)


def _plane_indices(z_eval: torch.Tensor, z_target: torch.Tensor) -> np.ndarray:
    idx = []
    for z_i in z_target:
        idx.append(int(torch.argmin(torch.abs(z_eval - z_i)).item()))
    return np.asarray(idx, dtype=np.int64)


def _summarize_scan(
    rho_bar: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_dists: torch.Tensor,
    focusing_mask: torch.Tensor,
    z_eval: torch.Tensor,
    z_target: torch.Tensor,
    crop_width: int,
    center_offsets,
) -> dict[str, np.ndarray]:
    I_lambda, onaxis, inmask = scan_cascade_intensity_z(
        rho_bar,
        sim_params,
        elem_params,
        z_dists,
        z_eval,
        mask=focusing_mask,
        center_offsets=center_offsets,
    )
    plane_idx = _plane_indices(z_eval, z_target)
    n_planes = int(z_target.shape[0])
    n_wvl, _, nx = I_lambda.shape
    half = int(crop_width) // 2
    cx = nx // 2
    profiles = np.zeros((n_planes, n_wvl, 2 * half), dtype=np.float32)
    crosstalk = np.zeros((n_planes, n_wvl), dtype=np.float32)
    for i, iz in enumerate(plane_idx):
        profiles[i] = I_lambda[:, iz, cx - half : cx + half].detach().cpu().numpy()
        crosstalk[i] = inmask[:, iz].detach().cpu().numpy()
    z_focus = z_eval[torch.argmax(inmask, dim=1)].detach().cpu().numpy()
    return {
        "onaxis": onaxis.detach().cpu().numpy().astype(np.float32),
        "inmask": inmask.detach().cpu().numpy().astype(np.float32),
        "profiles": profiles,
        "crosstalk": crosstalk,
        "plane_idx": plane_idx,
        "z_focus": z_focus.astype(np.float64),
    }


def main() -> None:
    args = _parse_args()
    if args.nelem < 1:
        raise ValueError("--nelem must be >= 1")
    if args.n_wvl < 2:
        raise ValueError("--n-wvl must be >= 2 to prescribe distinct planes")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    material_map = MATERIAL_MAP
    gap_map = GAP_MAP_DEFAULT
    membrane_map = MEMBRANE_MAP_SI3N4

    N_wvl = int(args.n_wvl)
    central_energy_ev = CENTRAL_ENERGY_EV_DEFAULT
    bandwidth = float(args.bandwidth)
    Nx = NX_DEFAULT
    dx = DX_DEFAULT
    f = F_DEFAULT
    inter_elem_dist = INTER_ELEM_DIST_DEFAULT
    membrane_thickness = MEMBRANE_THICKNESS_DEFAULT
    element_thickness = ELEMENT_THICKNESS_DEFAULT
    min_feature_size = MIN_FEATURE_SIZE_DEFAULT
    Nelem = int(args.nelem)
    focusing_threshold = FOCUSING_THRESHOLD_DEFAULT
    crop_width = CROP_WIDTH_DEFAULT
    chromatic_half_width = float(args.chromatic_half_width)
    n_z_eval = int(args.n_z_eval)

    opt_params = {
        "Nelem": Nelem,
        "min_feature_size": min_feature_size,
        "epsilon": EPSILON_DEFAULT,
        "tolerance": TOLERANCE_DEFAULT,
        "param_tolerance": PARAM_TOLERANCE_DEFAULT,
        "max_eval": int(args.max_eval),
        "min_beta": MIN_BETA_DEFAULT,
        "constraint_fac": CONSTRAINT_FAC_DEFAULT,
        "P": P_DEFAULT,
        "constraint_method": CONSTRAINT_METHOD_DEFAULT,
        "constraint_aggregation": CONSTRAINT_AGGREGATION_DEFAULT,
        "morph_agg_beta": MORPH_AGG_BETA_DEFAULT,
        "morph_beta": MORPH_BETA_DEFAULT,
        "beta_schedule": list(BETA_SCHEDULE_DEFAULT),
    }

    script_start_time = console.script_start(_LOG)
    console.kv(_LOG, "Nelem", Nelem)
    console.kv(_LOG, "N_wvl", N_wvl)
    console.kv(_LOG, "bandwidth", bandwidth)
    console.kv(_LOG, "chromatic_half_width", chromatic_half_width)
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
    energies_ev = wavelengths_to_energy_ev(lams)

    Ncenter = int(2 * 1.22 * min_feature_size / dx)
    focusing_mask = torch.zeros(1, Nx, device=device)
    focusing_mask[0, Nx // 2 - Ncenter // 2 : Nx // 2 + Ncenter // 2] = 1.0

    elem_params = {
        "thickness": element_thickness,
        "elem_map": material_map,
        "gap_map": gap_map,
        "membrane_map": membrane_map,
        "membrane_thickness": membrane_thickness,
    }
    center_offsets = None

    z_dists = torch.ones(Nelem - 1, device=device, dtype=torch.float64) * inter_elem_dist
    z_dists = torch.cat((z_dists, torch.tensor([f], device=device, dtype=torch.float64)))
    z_last_per_wavelength = linear_energy_focal_shift(
        lams, f, chromatic_half_width
    ).to(device=device, dtype=torch.float64)

    z_grid = torch.linspace(
        f - 1.2 * chromatic_half_width,
        f + 1.2 * chromatic_half_width,
        steps=n_z_eval,
        device=device,
        dtype=torch.float64,
    )
    z_eval = torch.unique(torch.sort(torch.cat((z_grid, z_last_per_wavelength)))[0])

    fwd_single = (elem_params, focusing_mask, z_dists, center_offsets)
    fwd_chromatic = (
        elem_params,
        focusing_mask,
        z_dists,
        center_offsets,
        z_last_per_wavelength,
    )

    console.banner(_LOG, "run 1: single-plane spectrum focusing")
    opt_start = time.time()
    raw_single, obj_list_single, _, _, model_single = run_torch_optimization(
        sim_params,
        opt_params,
        fwd_single,
        objective_function=forward_model_N_elements_mask,
    )
    console.elapsed(_LOG, "single-plane optimization", time.time() - opt_start)
    rho_single = _project_design(raw_single, model_single, device)

    console.banner(_LOG, "run 2: prescribed chromatic focusing")
    opt_start = time.time()
    raw_chromatic, obj_list_chromatic, _, _, model_chromatic = run_torch_optimization(
        sim_params,
        opt_params,
        fwd_chromatic,
        objective_function=forward_model_N_elements_mask_chromatic,
    )
    console.elapsed(_LOG, "chromatic optimization", time.time() - opt_start)
    rho_chromatic = _project_design(raw_chromatic, model_chromatic, device)

    console.banner(_LOG, "axial wavelength scans")
    scan_single = _summarize_scan(
        rho_single,
        sim_params,
        elem_params,
        z_dists,
        focusing_mask,
        z_eval,
        z_last_per_wavelength,
        crop_width,
        center_offsets,
    )
    scan_chromatic = _summarize_scan(
        rho_chromatic,
        sim_params,
        elem_params,
        z_dists,
        focusing_mask,
        z_eval,
        z_last_per_wavelength,
        crop_width,
        center_offsets,
    )

    lam_center = lams[int(torch.argmax(weights).item())]
    fzp_half = torch.tensor(
        zp_init(lam_center, f, min_feature_size, 1, sim_params),
        dtype=torch.float64,
        device=device,
    )
    z_fzp = torch.tensor([f], device=device, dtype=torch.float64)
    scan_fzp = _summarize_scan(
        fzp_half,
        sim_params,
        elem_params,
        z_fzp,
        focusing_mask,
        z_eval,
        z_last_per_wavelength,
        crop_width,
        center_offsets,
    )

    e0 = float(central_energy_ev)
    fzp_theory_z = f * (energies_ev.detach().cpu().numpy() / e0)

    params_dict = {
        "Nx": int(Nx),
        "dx": float(dx),
        "N_wvl": int(N_wvl),
        "central_energy_ev": float(central_energy_ev),
        "bandwidth": float(bandwidth),
        "min_feature_size": float(min_feature_size),
        "f": float(f),
        "inter_elem_dist": float(inter_elem_dist),
        "membrane_thickness": float(membrane_thickness),
        "element_thickness": float(element_thickness),
        "material": MATERIAL_DEFAULT,
        "Nelem": int(Nelem),
        "focusing_threshold": float(focusing_threshold),
        "crop_width": int(crop_width),
        "chromatic_half_width": float(chromatic_half_width),
        "n_z_eval": int(n_z_eval),
        "run1_description": "single-plane polychromatic focusing",
        "run2_description": "per-wavelength last hop, linear in energy",
        "optimizer": "run_torch_optimization",
        "forward_model_single": "forward_model_N_elements_mask",
        "forward_model_chromatic": "forward_model_N_elements_mask_chromatic",
        "opt_params": {
            "epsilon": float(EPSILON_DEFAULT),
            "tolerance": float(TOLERANCE_DEFAULT),
            "param_tolerance": float(PARAM_TOLERANCE_DEFAULT),
            "max_eval": int(args.max_eval),
            "min_beta": float(MIN_BETA_DEFAULT),
            "constraint_fac": float(CONSTRAINT_FAC_DEFAULT),
            "P": int(P_DEFAULT),
            "constraint_method": CONSTRAINT_METHOD_DEFAULT,
            "constraint_aggregation": CONSTRAINT_AGGREGATION_DEFAULT,
            "morph_beta": float(MORPH_BETA_DEFAULT),
            "morph_agg_beta": float(MORPH_AGG_BETA_DEFAULT),
        },
    }
    np.save(f"{save_dir}/{SAVE_PREFIX}_params_{save_time}.npy", params_dict)

    x_full = sim_params.x.detach().cpu().numpy()
    half = int(crop_width) // 2
    cx = int(Nx) // 2
    x_crop = x_full[cx - half : cx + half]

    save_sweep_results(
        f"{save_dir}/{SAVE_PREFIX}_results_{save_time}.npz",
        {
            "lams": lams.detach().cpu().numpy(),
            "weights": weights.detach().cpu().numpy(),
            "energies_ev": energies_ev.detach().cpu().numpy(),
            "z_dists": z_dists.detach().cpu().numpy(),
            "z_last_per_wavelength": z_last_per_wavelength.detach().cpu().numpy(),
            "z_eval": z_eval.detach().cpu().numpy(),
            "fzp_theory_z": np.asarray(fzp_theory_z, dtype=np.float64),
            "mask": focusing_mask.detach().cpu().numpy(),
            "x_crop": x_crop.astype(np.float64),
            "rho_bar_single": pack_binary_density(rho_single.detach().cpu().numpy()),
            "rho_bar_chromatic": pack_binary_density(rho_chromatic.detach().cpu().numpy()),
            "obj_list_single": np.asarray(obj_list_single, dtype=np.float32),
            "obj_list_chromatic": np.asarray(obj_list_chromatic, dtype=np.float32),
            "onaxis_single": scan_single["onaxis"],
            "onaxis_chromatic": scan_chromatic["onaxis"],
            "onaxis_fzp": scan_fzp["onaxis"],
            "inmask_single": scan_single["inmask"],
            "inmask_chromatic": scan_chromatic["inmask"],
            "inmask_fzp": scan_fzp["inmask"],
            "profiles_single": scan_single["profiles"],
            "profiles_chromatic": scan_chromatic["profiles"],
            "profiles_fzp": scan_fzp["profiles"],
            "crosstalk_single": scan_single["crosstalk"],
            "crosstalk_chromatic": scan_chromatic["crosstalk"],
            "crosstalk_fzp": scan_fzp["crosstalk"],
            "plane_idx": scan_single["plane_idx"],
            "z_focus_single": scan_single["z_focus"],
            "z_focus_chromatic": scan_chromatic["z_focus"],
            "z_focus_fzp": scan_fzp["z_focus"],
        },
    )
    console.file_saved(_LOG, f"{save_dir}/{SAVE_PREFIX}_results_{save_time}.npz")
    console.script_done(_LOG, script_start_time)


if __name__ == "__main__":
    main()
