import torch
import numpy as np
import time

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util import (
    create_material_map,
    gaussian_energy_spectrum,
    get_formatted_datetime,
    airy_1d_intensity,
    compute_opt_and_fzp_metrics_2d,
)  # type: ignore
from src.simparams import SimParams
from src.forwardmodels import (
    forward_model_N_elements_mask,
    forward_model_N_elements_mask_partial_coherence,
    forward_model_N_elements_mask_2d,
    forward_model_N_elements_mask_partial_coherence_2d,
)
from src.inversedesign_utils import zp_init
from src.optimizer import run_torch_optimization
from src import console

_LOG = "xray_focusing_partial_coherence"
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

# --- Partial coherence parameters ---
# sigma_g: RMS coherence length (m). The key parameter controlling spatial
#          coherence.  Larger -> more coherent (fewer modes carry weight).
#          Set to a large value (>> Nx*dx) to recover the fully coherent limit.
sigma_g = Nx * dx / 2 # half the aperture width — moderate partial coherence
# sigma_s: RMS beam width (m). Controls the Gaussian illumination envelope.
#          Kept large relative to the aperture for near-uniform illumination.
sigma_s = Nx * dx
# n_modes: number of Hermite-Gaussian coherent modes to use in the 1D partial-
# coherence objective. The 2D post-processing uses the separable outer-product
# of these modes in x and y.
n_modes = 15

center_offsets = None

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
    "sigma_s": sigma_s,
    "sigma_g": sigma_g,
    "n_modes": n_modes,
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

airy_mask = torch.tensor(
    airy_1d_intensity(
        sim_params.x[sim_params.x.shape[0] // 2 :].cpu().numpy(),
        1.22 * min_feature_size,
    ),
    device=device,
    dtype=torch.float64,
)
airy_mask = torch.cat((torch.flip(airy_mask, dims=(0,)), airy_mask))
airy_mask = airy_mask.view(1, -1)

mask = focusing_mask

z_dists = (Nelem - 1) * (inter_elem_dist,) + (f,)
z_dists = torch.tensor(z_dists, device=device, dtype=torch.float64)
center_offsets = None

fwd_model_args = (elem_params, mask, z_dists, center_offsets)

if __name__ == "__main__":
    script_start_time = console.script_start(_LOG)
    console.kv(_LOG, "Nelem", Nelem)
    console.kv(_LOG, "sigma_s", sigma_s)
    console.kv(_LOG, "sigma_g", sigma_g)
    console.kv(_LOG, "n_modes", n_modes)
    save_time = get_formatted_datetime()
    save_dir = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "outputs")
    os.makedirs(save_dir, exist_ok=True)
    console.banner(_LOG, "partial-coherence optimization")
    opt_start_time = time.time()
    raw_design, obj_list, intensity_list, extra_list, model = run_torch_optimization(
        sim_params,
        opt_params,
        fwd_model_args,
        objective_function=forward_model_N_elements_mask_partial_coherence,
    )
    console.elapsed(_LOG, "optimization", time.time() - opt_start_time)

    x_tensor = torch.tensor(raw_design, dtype=torch.float64)
    rho_tilde, _ = model.filter_density(x_tensor)
    rho_bar = (rho_tilde > 0.5).to(dtype=float)

    pc_metrics = compute_opt_and_fzp_metrics_2d(
        rho_bar,
        sim_params,
        fwd_model_args,
        min_feature_size=min_feature_size,
        focusing_threshold=focusing_threshold,
        crop_width=crop_width,
        forward_model_1d=forward_model_N_elements_mask_partial_coherence,
        forward_model_2d=forward_model_N_elements_mask_partial_coherence_2d,
        zp_init_func=zp_init,
    )
    coh_metrics = compute_opt_and_fzp_metrics_2d(
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

    opt_obj_pc_val = pc_metrics["opt_final_obj"]
    opt_I_pc_np = pc_metrics["opt_intensity_1d"]
    opt_width_pc = pc_metrics["opt_width"]
    opt_eff_pc = pc_metrics["opt_efficiency"]

    opt_obj_coh_val = coh_metrics["opt_final_obj"]
    opt_I_coh_np = coh_metrics["opt_intensity_1d"]
    opt_width_coh = coh_metrics["opt_width"]
    opt_eff_coh = coh_metrics["opt_efficiency"]

    fzp_obj_pc = pc_metrics["fzp_final_obj"]
    fzp_I_pc_np = pc_metrics["fzp_intensity_1d"]
    fzp_width_pc = pc_metrics["fzp_width"]
    fzp_eff_pc = pc_metrics["fzp_efficiency"]

    fzp_obj_coh = coh_metrics["fzp_final_obj"]
    fzp_I_coh_np = coh_metrics["fzp_intensity_1d"]
    fzp_width_coh = coh_metrics["fzp_width"]
    fzp_eff_coh = coh_metrics["fzp_efficiency"]
    fzp_x = pc_metrics["fzp_x"]

    console.banner(_LOG, "metrics: partial coherence vs coherent")
    console.info(
        _LOG,
        f"opt PC  obj={opt_obj_pc_val:.6f} width={opt_width_pc} eff={opt_eff_pc:.6f}",
    )
    console.info(
        _LOG,
        f"opt coh obj={opt_obj_coh_val:.6f} width={opt_width_coh} eff={opt_eff_coh:.6f}",
    )
    console.info(
        _LOG,
        f"fzp PC  obj={float(fzp_obj_pc):.6f} width={fzp_width_pc} eff={fzp_eff_pc:.6f}",
    )
    console.info(
        _LOG,
        f"fzp coh obj={float(fzp_obj_coh):.6f} width={fzp_width_coh} eff={fzp_eff_coh:.6f}",
    )

    # --- Save ---
    obj_list_np = np.array([float(o) if hasattr(o, 'item') else float(o) for o in obj_list])

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
        "crop_width": int(crop_width),
        "sigma_s": float(sigma_s),
        "sigma_g": float(sigma_g),
        "n_modes": int(n_modes),
        "partial_coherence_2d_model": "separable_gsm_outer_product",
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
    np.save(
        f"{save_dir}/xray_focusing_partial_coherence_params_{save_time}.npy",
        params_dict,
    )

    mask_save = mask.cpu().numpy()
    z_dists_save = z_dists.cpu().numpy()

    np.savez_compressed(
        f"{save_dir}/xray_focusing_partial_coherence_results_{save_time}.npz",
        opt_intensity_pc=opt_I_pc_np,
        opt_intensity_coh=opt_I_coh_np,
        opt_efficiency_pc=float(opt_eff_pc),
        opt_efficiency_coh=float(opt_eff_coh),
        opt_obj_pc=opt_obj_pc_val,
        opt_obj_coh=opt_obj_coh_val,
        opt_width_pc=float(opt_width_pc),
        opt_width_coh=float(opt_width_coh),
        rho_bar=pack_binary_density(rho_bar.detach().cpu().numpy()),
        fzp_intensity_pc=fzp_I_pc_np,
        fzp_intensity_coh=fzp_I_coh_np,
        fzp_efficiency_pc=float(fzp_eff_pc),
        fzp_efficiency_coh=float(fzp_eff_coh),
        fzp_obj_pc=float(fzp_obj_pc),
        fzp_obj_coh=float(fzp_obj_coh),
        fzp_width_pc=float(fzp_width_pc),
        fzp_width_coh=float(fzp_width_coh),
        fzp_x=fzp_x.cpu().numpy(),
        obj_list=obj_list_np,
        mask=mask_save,
        z_dists=z_dists_save,
    )

    console.file_saved(_LOG, f"{save_dir}/xray_focusing_partial_coherence_results_{save_time}.npz")
    console.script_done(_LOG, script_start_time)
