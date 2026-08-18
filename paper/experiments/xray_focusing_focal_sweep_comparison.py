import torch
import numpy as np
import time

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.util import (
    create_material_map,
    gaussian_energy_spectrum,
    width_central_peak,
    compute_opt_and_fzp_metrics_2d,
)
from src.util import get_formatted_datetime  # type: ignore
from src.simparams import SimParams
from src.forwardmodels import (
    forward_model_N_elements_mask_multi_z,
    forward_model_N_elements_mask,
    forward_model_N_elements_mask_2d,
)
from src.inversedesign_utils import zp_init
from src.optimizer import run_torch_optimization
from src import console

_LOG = "focal_sweep_comparison"
from paper.sweeps.density_io import save_sweep_results
from paper.sweeps.standard_params import (
    MATERIAL_DEFAULT,
    MATERIAL_MAP_AU,
    MATERIAL_MAP_NI,
    GAP_MAP_DEFAULT,
    MEMBRANE_MAP_SI3N4,
    N_WVL_DEFAULT,
    CENTRAL_ENERGY_EV_DEFAULT,
    BANDWIDTH_DEFAULT,
    F_DEFAULT,
    NX_DEFAULT,
    DX_DEFAULT,
    INTER_ELEM_DIST_DEFAULT,
    MEMBRANE_THICKNESS_DEFAULT,
    ELEMENT_THICKNESS_DEFAULT,
    MIN_FEATURE_SIZE_DEFAULT,
    N_ELEMENTS_DEFAULT,
    FOCUSING_THRESHOLD_DEFAULT,
    EPSILON_DEFAULT,
    TOLERANCE_DEFAULT,
    PARAM_TOLERANCE_DEFAULT,
    MIN_BETA_DEFAULT,
    CONSTRAINT_FAC_DEFAULT,
    P_DEFAULT,
    CONSTRAINT_METHOD_DEFAULT,
    CONSTRAINT_AGGREGATION_DEFAULT,
    MORPH_BETA_DEFAULT,
    MORPH_AGG_BETA_DEFAULT,
    MAX_EVAL_DEFAULT,
    CROP_WIDTH_DEFAULT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

material_map = MATERIAL_MAP_NI
gap_map = GAP_MAP_DEFAULT
membrane_map = MEMBRANE_MAP_SI3N4

N_wvl = N_WVL_DEFAULT
central_energy_ev = CENTRAL_ENERGY_EV_DEFAULT
bandwidth = BANDWIDTH_DEFAULT

# Script-specific overrides
Nx = NX_DEFAULT
dx = DX_DEFAULT
f = F_DEFAULT
inter_elem_dist = INTER_ELEM_DIST_DEFAULT
membrane_thickness = MEMBRANE_THICKNESS_DEFAULT
element_thickness = ELEMENT_THICKNESS_DEFAULT
min_feature_size = MIN_FEATURE_SIZE_DEFAULT
Nelem = 20
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

# Half-width of the focal sweep (distance from f to extreme planes). Last entries span f ± focal_sweep_half_width.
focal_sweep_half_width = 770e-6  # meters
# Number of z-distance tensors for the second optimization (sweep); center tensor is at f.
n_sweep_tensors = 11

center_offsets = None

lams, weights = gaussian_energy_spectrum(
    central_energy_ev=central_energy_ev,
    N=N_wvl,
    bandwidth=bandwidth,
    device=device,
    bandwidth_in_wavelength=False
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
    "membrane_thickness": membrane_thickness
}

opt_params = {
    "Nelem": Nelem,
    "min_feature_size": min_feature_size,
    "epsilon": epsilon,
    "tolerance": tolerance,
    "param_tolerance": param_tolerance,
    "max_eval": max_eval,
    "min_beta": min_beta,
    "constraint_fac": constraint_fac,
    "P": P,
    "constraint_method": constraint_method,
    "constraint_aggregation": constraint_aggregation,
    "morph_agg_beta": morph_agg_beta,
    "morph_beta": morph_beta,
}

Ncenter = int(2 * 1.22 * min_feature_size / dx)
focusing_mask = torch.zeros(1, Nx, device=device)
focusing_mask[0, Nx // 2 - Ncenter // 2 : Nx // 2 + Ncenter // 2] = 1.0


def build_z_single():
    """Single z-distance tensor: (Nelem-1)*inter_elem_dist then f."""
    z_dists = (Nelem - 1) * (inter_elem_dist,) + (f,)
    return torch.tensor(z_dists, device=device, dtype=torch.float64)


def build_z_distances_set_sweep(n_tensors: int | None = None):
    """Build n_tensors z-distance tensors; last entry sweeps evenly from f - focal_sweep_half_width to f + focal_sweep_half_width. Center tensor (index (n_tensors-1)//2) is at f. If n_tensors is None, uses global n_sweep_tensors."""
    if n_tensors is None:
        n_tensors = n_sweep_tensors
    z_base = build_z_single()
    if n_tensors <= 0:
        raise ValueError("n_tensors must be positive")
    if n_tensors == 1:
        last_entries = [f]
    else:
        # Evenly spaced from f - focal_sweep_half_width to f + focal_sweep_half_width
        last_entries = [
            f - focal_sweep_half_width + i * (2.0 * focal_sweep_half_width) / (n_tensors - 1)
            for i in range(n_tensors)
        ]
    z_set = []
    for last_z in last_entries:
        z_t = z_base.clone()
        z_t[-1] = last_z
        z_set.append(z_t)
    return z_set

def compute_zp_x():
    """Compute zone plate parameters."""
    return torch.tensor(zp_init(lams[lams.argmax()], f, min_feature_size, 1, sim_params))

def forward_model_multi_z_single_plane(
    rho_bar: torch.Tensor,
    sim_params_1d: SimParams,
    elem_params_local,
    mask_local,
    z_dists_local,
    center_offsets_local,
):
    """
    Adapter that evaluates the multi-z 1D forward model on a single z-distance
    tensor by wrapping it in a length-1 list with unit weight.
    """
    z_list = [z_dists_local]
    z_weights = torch.tensor([1.0], device=sim_params_1d.device, dtype=torch.float64)
    fwd_args = (elem_params_local, mask_local, z_list, center_offsets_local, z_weights)
    return forward_model_N_elements_mask_multi_z(rho_bar, sim_params_1d, *fwd_args)


def zp_init_fixed_design(lam_center, f_scalar, min_feature_size_local, one, sim_params_local):
    """
    Adapter for use with compute_opt_and_fzp_metrics_2d inside this script:
    always returns the precomputed 1D FZP profile so that the zone-plate
    design is independent of the evaluation plane z.
    """
    return compute_zp_x().detach().cpu().numpy()


if __name__ == "__main__":
    script_start_time = console.script_start(_LOG)
    console.kv(_LOG, "Nelem", Nelem)
    console.kv(_LOG, "n_sweep_tensors", n_sweep_tensors)
    save_time = get_formatted_datetime()
    save_dir = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "outputs")
    os.makedirs(save_dir, exist_ok=True)

    # ----- Run 1: single z at focal length -----
    z_single = build_z_single()
    fwd_model_args_run1 = (elem_params, focusing_mask, [z_single], center_offsets, [1.0])

    console.banner(_LOG, "run 1: single focal plane")
    opt_start_time = time.time()
    raw_design_run1, obj_list_run1, intensity_list_run1, extra_list_run1, model_run1 = run_torch_optimization(
        sim_params,
        opt_params,
        fwd_model_args_run1,
        objective_function=forward_model_N_elements_mask_multi_z,
    )
    console.elapsed(_LOG, "run 1 optimization", time.time() - opt_start_time)

    x_tensor_run1 = torch.tensor(raw_design_run1, dtype=torch.float64, device=device)
    rho_tilde_run1, _ = model_run1.filter_density(x_tensor_run1)
    rho_bar_run1 = (rho_tilde_run1 > 0.5).to(dtype=torch.float64)

    # Evaluate optimized design and FZP baseline at the focal plane using the
    # shared 2D helper with a single-z adapter around the multi-z forward model.
    fwd_model_args_run1_eval = (elem_params, focusing_mask, z_single, center_offsets)
    metrics_run1 = compute_opt_and_fzp_metrics_2d(
        rho_bar_run1,
        sim_params,
        fwd_model_args_run1_eval,
        min_feature_size=min_feature_size,
        focusing_threshold=focusing_threshold,
        crop_width=int(crop_width),
        forward_model_1d=forward_model_multi_z_single_plane,
        forward_model_2d=forward_model_N_elements_mask_2d,
        zp_init_func=zp_init_fixed_design,
    )

    opt_final_obj_run1 = metrics_run1["opt_final_obj"]
    opt_intensity_run1_center = metrics_run1["opt_intensity_1d"]
    opt_width_run1 = metrics_run1["opt_width"]
    opt_efficiency_run1 = metrics_run1["opt_efficiency"]

    obj_list_run1_np = np.array([float(o) if hasattr(o, 'item') else float(o) for o in obj_list_run1])

    # ----- Run 2: n_sweep_tensors z tensors sweeping around focal length -----
    z_distances_set_run2 = build_z_distances_set_sweep()
    z_weights_run2 = list(np.ones(len(z_distances_set_run2)))  # equal weights
    z_weights_run2 = torch.tensor(z_weights_run2, device=device, dtype=torch.float64)
    fwd_model_args_run2 = (elem_params, focusing_mask, z_distances_set_run2, center_offsets, z_weights_run2)

    console.banner(_LOG, f"run 2: {n_sweep_tensors} focal planes in objective")
    opt_start_time = time.time()
    raw_design_run2, obj_list_run2, intensity_list_run2, extra_list_run2, model_run2 = run_torch_optimization(
        sim_params,
        opt_params,
        fwd_model_args_run2,
        objective_function=forward_model_N_elements_mask_multi_z,
    )
    console.elapsed(_LOG, "run 2 optimization", time.time() - opt_start_time)

    x_tensor_run2 = torch.tensor(raw_design_run2, dtype=torch.float64, device=device)
    rho_tilde_run2, _ = model_run2.filter_density(x_tensor_run2)
    rho_bar_run2 = (rho_tilde_run2 > 0.5).to(dtype=torch.float64)

    # Evaluate run 2 optimized design at the focal plane using the same
    # single-z 2D objective as run 1 (z = f).
    metrics_run2 = compute_opt_and_fzp_metrics_2d(
        rho_bar_run2,
        sim_params,
        fwd_model_args_run1_eval,
        min_feature_size=min_feature_size,
        focusing_threshold=focusing_threshold,
        crop_width=int(crop_width),
        forward_model_1d=forward_model_multi_z_single_plane,
        forward_model_2d=forward_model_N_elements_mask_2d,
        zp_init_func=zp_init_fixed_design,
        compute_fzp=False,
    )

    opt_final_obj_run2 = metrics_run2["opt_final_obj"]
    opt_intensity_run2_center = metrics_run2["opt_intensity_1d"]

    obj_list_run2_np = np.array([float(o) if hasattr(o, 'item') else float(o) for o in obj_list_run2])

    # ----- Zone plate reference intensity at focal plane -----
    zp_intensity = metrics_run1["fzp_intensity_1d"]
    zp_width = metrics_run1["fzp_width"]
    zp_efficiency = metrics_run1["fzp_efficiency"]

    # Precompute FZP design for saving (matches zp_init_fixed_design profile).
    zp_x_tensor = compute_zp_x().to(device=device, dtype=torch.float64)

    # ----- 2D center-slice intensities vs z after optimization -----
    # Define evaluation z distances around the focal length (independent of multi-z objective grids)
    Nz_eval = 101
    z_eval_half_width = 1.2e-3 # meters
    z_eval = torch.linspace(
        f - z_eval_half_width,
        f + z_eval_half_width,
        steps=Nz_eval,
        device=device,
        dtype=torch.float64,
    )

    # Allocate (Nx, Nz_eval) arrays for zone plate, run 1, and run 2 center slices
    zp_intensity_z_sweep = np.zeros((Nx, Nz_eval), dtype=np.float64)
    opt_intensity_run1_z_sweep = np.zeros((Nx, Nz_eval), dtype=np.float64)
    opt_intensity_run2_z_sweep = np.zeros((Nx, Nz_eval), dtype=np.float64)

    # Precompute base z-distance tensor for cascaded designs (Nelem-1)*inter_elem_dist + variable final distance
    z_base_cascade = build_z_single()

    for idx_z, z_val in enumerate(z_eval):
        # Build a single-z distance tensor for this evaluation plane
        z_run = z_base_cascade.clone()
        z_run[-1] = z_val
        fwd_args_z = (elem_params, focusing_mask, z_run, center_offsets)

        # Zone plate at this z (using the shared 2D helper with fixed FZP design)
        metrics_z_run1 = compute_opt_and_fzp_metrics_2d(
            rho_bar_run1,
            sim_params,
            fwd_args_z,
            min_feature_size=min_feature_size,
            focusing_threshold=focusing_threshold,
            crop_width=int(crop_width),
            forward_model_1d=forward_model_multi_z_single_plane,
            forward_model_2d=forward_model_N_elements_mask_2d,
            zp_init_func=zp_init_fixed_design,
        )
        zp_center_row = metrics_z_run1["fzp_intensity_1d"]
        run1_center_row = metrics_z_run1["opt_intensity_1d"]

        zp_intensity_z_sweep[:, idx_z] = zp_center_row
        opt_intensity_run1_z_sweep[:, idx_z] = run1_center_row

        # Run 2 optimized cascade at this z (same base inter-element pattern, variable final distance)
        metrics_z_run2 = compute_opt_and_fzp_metrics_2d(
            rho_bar_run2,
            sim_params,
            fwd_args_z,
            min_feature_size=min_feature_size,
            focusing_threshold=focusing_threshold,
            crop_width=int(crop_width),
            forward_model_1d=forward_model_multi_z_single_plane,
            forward_model_2d=forward_model_N_elements_mask_2d,
            zp_init_func=zp_init_fixed_design,
            compute_fzp=False,
        )
        run2_center_row = metrics_z_run2["opt_intensity_1d"]
        opt_intensity_run2_z_sweep[:, idx_z] = run2_center_row

    # ----- Save params -----
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
        "material": "ni",
        "Nelem": int(Nelem),
        "focusing_threshold": float(focusing_threshold),
        "focal_sweep_half_width": float(focal_sweep_half_width),
        "n_sweep_tensors": int(n_sweep_tensors),
        "run1_description": "single z at focal length",
        "run2_description": f"{n_sweep_tensors} z tensors, last entry sweep around f, equal weights, same mask",
        "optimizer": "run_torch_optimization",
        "forward_model": "forward_model_N_elements_mask_multi_z",
        "opt_params": {
            "epsilon": float(epsilon),
            "tolerance": float(tolerance),
            "param_tolerance": float(param_tolerance),
            "max_eval": int(max_eval),
            "min_beta": float(min_beta),
            "constraint_fac": float(constraint_fac),
            "P": int(P),
            "constraint_method": constraint_method,
            "constraint_aggregation": constraint_aggregation,
            "morph_beta": float(morph_beta),
            "morph_agg_beta": float(morph_agg_beta),
        },
    }
    np.save(f"{save_dir}/fig4b_xray_focusing_focal_sweep_comparison_params_{save_time}.npy", params_dict)

    # ----- Save results -----
    z_distances_run1_np = z_single.cpu().numpy()
    z_distances_set_run2_np = np.stack([z.cpu().numpy() for z in z_distances_set_run2])
    mask_save = focusing_mask.cpu().numpy()

    save_sweep_results(
        f"{save_dir}/fig4b_xray_focusing_focal_sweep_comparison_results_{save_time}.npz",
        {
            "opt_intensity": opt_intensity_run1_center,
            "opt_efficiency": float(opt_efficiency_run1),
            "opt_obj": float(opt_final_obj_run1),
            "opt_width": float(opt_width_run1),
            "rho_bar": rho_bar_run1.detach().cpu().numpy(),
            "obj_list": obj_list_run1_np,
            "z_distances_run1": z_distances_run1_np,
            "opt_intensity_run2_center": opt_intensity_run2_center,
            "rho_bar_run2": rho_bar_run2.detach().cpu().numpy(),
            "obj_list_run2": obj_list_run2_np,
            "z_distances_set_run2": z_distances_set_run2_np,
            "opt_obj_run2": float(opt_final_obj_run2),
            "zp_intensity": zp_intensity,
            "zp_x": zp_x_tensor.cpu().numpy(),
            "zp_width": float(zp_width),
            "zp_efficiency": float(zp_efficiency),
            "mask": mask_save,
            "z_eval": z_eval.cpu().numpy(),
            "zp_intensity_z_sweep": zp_intensity_z_sweep,
            "opt_intensity_run1_z_sweep": opt_intensity_run1_z_sweep,
            "opt_intensity_run2_z_sweep": opt_intensity_run2_z_sweep,
        },
    )

    console.file_saved(_LOG, f"{save_dir}/fig4b_xray_focusing_focal_sweep_comparison_results_{save_time}.npz")
    console.script_done(_LOG, script_start_time)
