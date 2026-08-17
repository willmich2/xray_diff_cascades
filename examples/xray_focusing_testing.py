import torch
import numpy as np
import time

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util import (
    gaussian_energy_spectrum,
    get_formatted_datetime,
    airy_1d_intensity,
    compute_opt_and_fzp_metrics_2d,
)  # type: ignore
from src.simparams import SimParams
from src.forwardmodels import forward_model_N_elements_mask, forward_model_N_elements_mask_2d
from src.inversedesign_utils import zp_init
from src.optimizer import run_torch_optimization
from src import console

_LOG = "xray_focusing_testing"
from paper.sweeps.density_io import pack_binary_density
from paper.sweeps.standard_params import (
    MATERIAL_DEFAULT,
    MATERIAL_MAP,
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

material_map = MATERIAL_MAP
gap_map = GAP_MAP_DEFAULT
membrane_map = MEMBRANE_MAP_SI3N4

N_wvl = N_WVL_DEFAULT
central_energy_ev = CENTRAL_ENERGY_EV_DEFAULT
bandwidth = BANDWIDTH_DEFAULT

# Script-specific overrides. These values keep the example short; use the
# paper defaults in `paper/sweeps/standard_params.py` for production runs.
N_wvl = N_WVL_DEFAULT
Nx = NX_DEFAULT
dx = DX_DEFAULT
f = F_DEFAULT
inter_elem_dist = INTER_ELEM_DIST_DEFAULT
membrane_thickness = MEMBRANE_THICKNESS_DEFAULT
element_thickness = ELEMENT_THICKNESS_DEFAULT
min_feature_size = MIN_FEATURE_SIZE_DEFAULT
Nelem = N_ELEMENTS_DEFAULT
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

center_offsets = None

lams, weights = gaussian_energy_spectrum(
        central_energy_ev = central_energy_ev, 
        N = N_wvl, 
        bandwidth = bandwidth, 
        device = device, 
        bandwidth_in_wavelength = False
    )

sim_params = SimParams(
    Ny=1, 
    Nx=Nx, 
    dx=dx,
    device=device, 
    dtype = torch.complex128,
    lams=lams, 
    weights=weights
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

Ncenter = int(2*1.22*min_feature_size / dx)
focusing_mask = torch.zeros(1, Nx, device=device)
focusing_mask[0, Nx//2 - Ncenter//2:Nx//2 + Ncenter//2] = 1.0

airy_mask = torch.tensor(
    airy_1d_intensity(sim_params.x[sim_params.x.shape[0]//2:].cpu().numpy(), 1.22*min_feature_size), 
    device=device, 
    dtype=torch.float64
    )
airy_mask = torch.cat((torch.flip(airy_mask, dims=(0,)), airy_mask))
airy_mask = airy_mask.view(1, -1)

mask = focusing_mask

z_dists = (Nelem - 1)*(inter_elem_dist,) + (f,)
z_dists = torch.tensor(z_dists, device=device, dtype=torch.float64)
center_offsets = None

fwd_model_args = (elem_params, mask, z_dists, center_offsets)

if __name__ == "__main__":
    script_start_time = console.script_start(_LOG)
    console.kv(_LOG, "Nelem", Nelem)
    console.kv(_LOG, "Nx", Nx)
    console.kv(_LOG, "device", device)
    save_time = get_formatted_datetime()
    save_dir = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "outputs")
    os.makedirs(save_dir, exist_ok=True)
    console.info(_LOG, f"output directory {save_dir}")

    console.banner(_LOG, "topology optimization")
    opt_start_time = time.time()
    raw_design, obj_list, intensity_list, extra_list, model = run_torch_optimization(sim_params, opt_params, fwd_model_args)
    console.elapsed(_LOG, "optimization", time.time() - opt_start_time)

    x_tensor = torch.tensor(raw_design, dtype=torch.float64)

    rho_tilde, _ = model.filter_density(x_tensor)
    rho_bar = (rho_tilde > 0.5).to(dtype=float)

    console.banner(_LOG, "post-optimization metrics (opt vs FZP)")
    metrics = compute_opt_and_fzp_metrics_2d(
        rho_bar,
        sim_params,
        fwd_model_args,
        min_feature_size=min_feature_size,
        focusing_threshold=focusing_threshold,
        crop_width=crop_width,
        forward_model_1d=forward_model_N_elements_mask,
        forward_model_2d=forward_model_N_elements_mask_2d,
        zp_init_func=zp_init,
    )

    opt_final_obj = metrics["opt_final_obj"]
    opt_width = metrics["opt_width"]
    opt_efficiency = metrics["opt_efficiency"]
    opt_intensity_1d = metrics["opt_intensity_1d"]

    fzp_final_obj = metrics["fzp_final_obj"]
    fzp_width = metrics["fzp_width"]
    fzp_efficiency = metrics["fzp_efficiency"]
    fzp_intensity_1d = metrics["fzp_intensity_1d"]
    fzp_x = metrics["fzp_x"]
    console.info(
        _LOG,
        (
            f"opt obj={opt_final_obj:.6f} width={opt_width} eff={opt_efficiency:.6f} | "
            f"fzp obj={fzp_final_obj:.6f} width={fzp_width} eff={fzp_efficiency:.6f}"
        ),
    )

    # Convert obj_list and intensity_list to numpy for saving
    obj_list_np = np.array([float(o) if hasattr(o, 'item') else float(o) for o in obj_list])

    console.banner(_LOG, "saving results")
    params_dict = {
        "Nx": int(Nx),
        "dx": float(dx),
        "N_wvl": int(N_wvl),
        "central_energy_ev": float(central_energy_ev),
        "bandwidth": float(bandwidth),
        "min_feature_size": float(min_feature_size),
        "f": float(f),
        "membrane_thickness": float(membrane_thickness),
        "element_thickness": float(element_thickness),
        "material": MATERIAL_DEFAULT,
        "Nelem": int(Nelem),
        "focusing_threshold": float(focusing_threshold),
        "optimizer": "run_torch_optimization",
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
    np.save(f"{save_dir}/fig1d_xray_focusing_testing_params_{save_time}.npy", params_dict)
    
    # Save mask and z_dists
    mask_save = mask.cpu().numpy()
    z_dists_save = z_dists.cpu().numpy()
    
    np.savez_compressed(
        f"{save_dir}/fig1d_xray_focusing_testing_results_{save_time}.npz",
        opt_intensity=opt_intensity_1d,
        opt_efficiency=float(opt_efficiency),
        opt_obj=float(opt_final_obj),
        opt_width=float(opt_width),
        rho_bar=pack_binary_density(rho_bar.detach().cpu().numpy()),
        fzp_intensity=fzp_intensity_1d,
        fzp_efficiency=float(fzp_efficiency),
        fzp_obj=float(fzp_final_obj),
        fzp_width=float(fzp_width),
        fzp_x=fzp_x.cpu().numpy(),
        obj_list=obj_list_np,
        mask=mask_save,
        z_dists=z_dists_save
    )

    console.file_saved(_LOG, f"{save_dir}/fig1d_xray_focusing_testing_results_{save_time}.npz")
    console.script_done(_LOG, script_start_time)