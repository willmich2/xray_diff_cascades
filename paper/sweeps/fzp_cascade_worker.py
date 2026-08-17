"""Sweep worker: optimized cascade vs intermediate-field FZP cascade.

The returned ``fzp_*`` metrics are for an N-element coinciding-foci Fresnel
zone-plate cascade (not a single zone plate). Optimized-cascade metrics stay
in ``opt_*``. This matches the result schema used by ``_collect_results`` in
``sweep_framework.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src import console

# Default FZP stacking gap for intermediate-field coinciding-foci plates
# (Gleber et al., Opt. Express 2014: 0.3–1 mm at 10 keV). The optimized
# cascade keeps INTER_ELEM_DIST_DEFAULT (1 cm) unless overridden.
FZP_INTER_ELEM_DIST_DEFAULT = 1e-3


def coinciding_focal_lengths(nelem: int, f: float, inter_elem_dist: float) -> list[float]:
    return [f + (nelem - 1 - i) * inter_elem_dist for i in range(nelem)]


def fzp_cascade_vs_opt_worker(task: dict[str, Any]) -> dict[str, Any]:
    from paper.sweeps import sweep_framework
    from paper.sweeps.density_io import INTENSITY_DTYPE, pack_binary_density
    from paper.sweeps.sweep_framework import _method_from_model, _resolve_model
    from src.inversedesign_utils import fzp_cascade_half_profiles, zp_init
    from src.optimizer import run_torch_optimization
    from src.simparams import SimParams
    from src.util import compute_opt_and_fzp_metrics_2d, focusing_gain, gaussian_energy_spectrum

    gpu_id = sweep_framework._worker_gpu_id if sweep_framework._worker_gpu_id is not None else 0
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
    task_id = int(task.get("task_id", -1))
    console.info(
        "sweep.worker",
        (
            f"task {task_id} started on {device.type}"
            + (f":{gpu_id}" if device.type == "cuda" else "")
            + f" index={task.get('index')} axes={task.get('axis_values')}"
        ),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    params = task["params"]
    Nx = int(params["Nx"])
    dx = float(params["dx"])
    N_wvl = int(params["N_wvl"])
    Nelem = int(params["Nelem"])
    f = float(params["f"])
    opt_inter_elem_dist = float(params["inter_elem_dist"])
    fzp_inter_elem_dist = float(params.get("fzp_inter_elem_dist", FZP_INTER_ELEM_DIST_DEFAULT))
    min_feature_size = float(params["min_feature_size"])
    r_max = (Nx // 2) * dx

    lams, weights = gaussian_energy_spectrum(
        central_energy_ev=float(params["central_energy_ev"]),
        N=N_wvl,
        bandwidth=float(params["bandwidth"]),
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
    Ncenter = int(2 * 1.22 * min_feature_size / dx)
    focusing_mask = torch.zeros(1, Nx, device=device)
    focusing_mask[0, Nx // 2 - Ncenter // 2 : Nx // 2 + Ncenter // 2] = 1.0

    elem_params = {
        "thickness": float(params["element_thickness"]),
        "elem_map": params["material_map"],
        "gap_map": params["gap_map"],
        "membrane_map": params["membrane_map"],
        "membrane_thickness": float(params["membrane_thickness"]),
        "propagation_method": _method_from_model(str(params["optimization_model"])),
    }
    if params.get("sigma_s") is not None:
        elem_params["sigma_s"] = float(params["sigma_s"])
    if params.get("sigma_g") is not None:
        elem_params["sigma_g"] = float(params["sigma_g"])
    if params.get("n_modes") is not None:
        elem_params["n_modes"] = int(params["n_modes"])

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
    opt_fwd_model_args = (elem_params, focusing_mask, opt_z_dists, params.get("center_offsets"))
    fzp_fwd_model_args = (elem_params, focusing_mask, fzp_z_dists, params.get("center_offsets"))

    opt_params = {
        "Nelem": Nelem,
        "min_feature_size": min_feature_size,
        "epsilon": float(params["epsilon"]),
        "tolerance": float(params["tolerance"]),
        "param_tolerance": float(params["param_tolerance"]),
        "max_eval": int(params["max_eval"]),
        "min_beta": float(params["min_beta"]),
        "beta_schedule": list(params["beta_schedule"]),
        "constraint_fac": float(params["constraint_fac"]),
        "P": int(params["P"]),
        "constraint_method": str(params["constraint_method"]),
        "constraint_aggregation": str(params["constraint_aggregation"]),
        "morph_agg_beta": float(params["morph_agg_beta"]),
        "morph_beta": float(params["morph_beta"]),
    }

    objective_function = _resolve_model(str(params["optimization_model"]), dimension="opt")
    metric_1d = _resolve_model(str(params["metric_model_1d"]), dimension="metric_1d")
    metric_2d = _resolve_model(str(params["metric_model_2d"]), dimension="metric_2d")
    elem_params["propagation_method"] = _method_from_model(str(params["metric_model_1d"]))
    raw_design, obj_list, _intensity_list, _extra_list, model = run_torch_optimization(
        sim_params,
        opt_params,
        opt_fwd_model_args,
        objective_function=objective_function,
    )
    x_tensor = torch.tensor(raw_design, dtype=torch.float64, device=device)
    rho_tilde, _ = model.filter_density(x_tensor)
    rho_bar = (rho_tilde > 0.5).to(dtype=float)

    metric_kwargs = dict(
        min_feature_size=min_feature_size,
        focusing_threshold=float(params["focusing_threshold"]),
        crop_width=int(params["crop_width"]),
        forward_model_1d=metric_1d,
        forward_model_2d=metric_2d,
        zp_init_func=zp_init,
        compute_fzp=False,
    )
    opt_metrics = compute_opt_and_fzp_metrics_2d(
        rho_bar,
        sim_params,
        opt_fwd_model_args,
        **metric_kwargs,
    )

    lam_center = lams[lams.argmax()]
    focal_lengths = coinciding_focal_lengths(Nelem, f, fzp_inter_elem_dist)
    fzp_cascade_np, _fzp_f, _r_theory, _r_used = fzp_cascade_half_profiles(
        lam_center,
        focal_lengths,
        min_feature_size,
        sim_params,
        max_radius=r_max,
    )
    fzp_cascade_x = torch.tensor(fzp_cascade_np, dtype=torch.float64, device=device)
    cascade_metrics = compute_opt_and_fzp_metrics_2d(
        fzp_cascade_x,
        sim_params,
        fzp_fwd_model_args,
        **metric_kwargs,
    )

    opt_gain = focusing_gain(
        np.asarray(opt_metrics["opt_intensity_1d"]),
        float(params["focusing_threshold"]),
    )
    fzp_gain = focusing_gain(
        np.asarray(cascade_metrics["opt_intensity_1d"]),
        float(params["focusing_threshold"]),
    )
    obj_np = np.array(
        [float(o) if hasattr(o, "item") else float(o) for o in obj_list],
        dtype=np.float64,
    )
    console.info(
        "sweep.worker",
        (
            f"task {task_id} finished: opt_eff={float(opt_metrics['opt_efficiency']):.4f} "
            f"fzp_cascade_eff={float(cascade_metrics['opt_efficiency']):.4f} "
            f"opt_gain={opt_gain:.4f} fzp_cascade_gain={fzp_gain:.4f}"
        ),
    )
    return {
        "status": "ok",
        "index": tuple(task["index"]),
        "task_id": int(task["task_id"]),
        "result": {
            "rho_bar": pack_binary_density(rho_bar.detach().cpu().numpy()),
            "obj_list": obj_np,
            "opt_obj": float(opt_metrics["opt_final_obj"]),
            "opt_eff": float(opt_metrics["opt_efficiency"]),
            "opt_width": float(opt_metrics["opt_width"]),
            "opt_gain": float(opt_gain),
            "opt_intensity": np.asarray(opt_metrics["opt_intensity_1d"], dtype=INTENSITY_DTYPE),
            "fzp_obj": float(cascade_metrics["opt_final_obj"]),
            "fzp_eff": float(cascade_metrics["opt_efficiency"]),
            "fzp_width": float(cascade_metrics["opt_width"]),
            "fzp_gain": float(fzp_gain),
            "fzp_intensity": np.asarray(cascade_metrics["opt_intensity_1d"], dtype=INTENSITY_DTYPE),
        },
    }
