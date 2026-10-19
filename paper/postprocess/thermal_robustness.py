import os
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.util import (  # type: ignore
    create_material_map,
    gaussian_energy_spectrum,
    get_formatted_datetime,
    apply_thermal_imperfection,
    compute_opt_and_fzp_metrics_2d,
)
from src.simparams import SimParams  # type: ignore
from src.forwardmodels import forward_model_N_elements_mask, forward_model_N_elements_mask_2d  # type: ignore
from src.inversedesign_utils import zp_init  # type: ignore
from paper.data_paths import (
    DEFAULT_BASE_SWEEP_ID,
    DEFAULT_DATA_DIR,
    output_id_label,
    resolve_fig1_n_sweep_paths,
    robustness_results_path,
)
from paper.sweeps.density_io import density_half_profile, load_opt_rhos
from src import console

_LOG = "thermal_robustness"

DEFAULT_WORKERS_PER_GPU = int(os.environ.get("MAX_WORKERS", "3"))

_worker_gpu_id = None
_opt_rhos = None
_params = None
_sweep_arrs = None

# Thermal coefficients
au_cte = 1.4e-5
sin_cte = 3.3e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate thermal robustness from a fig1_N_sweeps run.")
    parser.add_argument(
        "--base-id",
        default=DEFAULT_BASE_SWEEP_ID,
        help="Timestamp for fig1_N_sweeps_* inputs; omit for bundled paper_data/ (no timestamp suffix).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run index (fig1_N_sweeps_results_run_<run_id>.npz or fig1_N_sweeps_results_<ID>_run_<run_id>.npz).",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing sweep/result files.")
    parser.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    return parser.parse_args()


def _init_worker(gpu_queue, results_path: str, params_path: str, sweep_arrays_path: str):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    global _opt_rhos, _params, _sweep_arrs

    # Load large, read-only inputs once per worker process.
    # IMPORTANT: do not keep NpzFile handles around (they are not picklable and
    # can hold open BufferedReader file descriptors); read arrays then close.
    with np.load(results_path, allow_pickle=True) as results_np:
        _opt_rhos = load_opt_rhos(results_np["opt_rhos"])
    _params = np.load(params_path, allow_pickle=True).item()
    _sweep_arrs = np.load(sweep_arrays_path, allow_pickle=True).item()


def worker(task):
    """
    One robustness evaluation for a single (choice_index, temp_index).
    Returns CPU scalars only, pickle-safe for ProcessPoolExecutor.
    """
    global _worker_gpu_id
    global _opt_rhos, _params, _sweep_arrs
    choice_idx, temp_idx, choices, temps = task
    ch = choices[choice_idx]
    console.info(
        _LOG,
        f"worker task choice={ch} temp_idx={temp_idx} delta_T={float(temps[temp_idx]):.1f} K",
    )

    gpu_id = _worker_gpu_id if _worker_gpu_id is not None else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    cuda_device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if _opt_rhos is None or _params is None or _sweep_arrs is None:
        raise RuntimeError("Worker globals not initialized. Did initializer run?")

    rhos = _opt_rhos
    params = _params
    sweep_arrs = _sweep_arrs
    Nelems = sweep_arrs["Nelems"]

    t = float(temps[temp_idx])

    Nx = int(params["Nx"])
    dx = float(params["dx"])
    f = float(params["f"])
    min_feature_size = float(params["min_feature_size"])
    focusing_threshold = float(params.get("focusing_threshold", 1e-2))

    crop_indices = int(params.get("crop_width", 256))
    N_trials = 1

    # spectrum
    lams, weights = gaussian_energy_spectrum(
        central_energy_ev=float(params["central_energy_ev"]),
        N=int(params["N_wvl"]),
        bandwidth=float(params["bandwidth"]),
        device=cuda_device,
        bandwidth_in_wavelength=False,
    )

    # SimParams for zp_init (1D) and forward model (2D)
    sim_params_1d = SimParams(
        Ny=1,
        Nx=Nx,
        dx=dx,
        device=cuda_device,
        dtype=torch.complex128,
        lams=lams,
        weights=weights,
    )
    sim_params_2d = SimParams(
        Ny=Nx,
        Nx=Nx,
        dx=dx,
        device=cuda_device,
        dtype=torch.complex128,
        lams=lams,
        weights=weights,
    )

    # focusing mask
    Ncenter = int(2 * 1.22 * min_feature_size / dx)
    mask = torch.zeros(1, Nx, device=cuda_device)
    mask[0, Nx // 2 - Ncenter // 2 : Nx // 2 + Ncenter // 2] = 1.0

    Nelem = int(Nelems[ch[1]])
    z_dists = torch.ones(Nelem - 1, device=cuda_device) * float(params["inter_elem_dist"])
    z_dists = torch.cat((z_dists, torch.tensor([f], device=cuda_device)))
    opt_x_base = density_half_profile(rhos, ch, int(Nelem * Nx // 2))

    fzp_efficiencies = np.zeros((N_trials,), dtype=float)
    opt_efficiencies = np.zeros((N_trials,), dtype=float)
    fzp_widths = np.zeros((N_trials,), dtype=float)
    opt_widths = np.zeros((N_trials,), dtype=float)

    for k in range(N_trials):
        # --- FZP baseline with thermal imperfection ---
        fzp_x_base = zp_init(lams[lams.argmax()], f, min_feature_size, 1, sim_params_1d)
        fzp_x_base_np = (
            fzp_x_base.detach().cpu().numpy() if isinstance(fzp_x_base, torch.Tensor) else np.asarray(fzp_x_base)
        )

        fzp_profile_full = np.concatenate((fzp_x_base_np, np.flip(fzp_x_base_np)))
        (
            fzp_x_thermal_full,
            fzp_thickness_scale_membrane,
            fzp_thickness_scale_grating,
        ) = apply_thermal_imperfection(
            fzp_profile_full,
            dt_max=t,
            cte_grating=au_cte,
            cte_membrane=sin_cte,
            profile_type="uniform",
        )
        fzp_x_thermal_half = fzp_x_thermal_full[: fzp_x_thermal_full.shape[0] // 2]
        fzp_x = torch.tensor(fzp_x_thermal_half, device=cuda_device, dtype=torch.float64)

        fzp_elem_params = {
            "thickness": fzp_thickness_scale_grating * float(params["element_thickness"]),
            "elem_map": create_material_map("au"),
            "gap_map": [np.array([0, np.inf]), np.array([1.0, 1.0])],
            "membrane_map": create_material_map("si3n4"),
            "membrane_thickness": fzp_thickness_scale_membrane * float(params["membrane_thickness"]),
        }
        fzp_z_dists = torch.tensor([f], device=cuda_device, dtype=torch.float64)
        fzp_centers = ((0.0, 0.0),)
        fzp_fwd_model_args = (fzp_elem_params, mask, fzp_z_dists, fzp_centers)

        fzp_metrics = compute_opt_and_fzp_metrics_2d(
            fzp_x,
            sim_params_1d,
            fzp_fwd_model_args,
            min_feature_size=min_feature_size,
            focusing_threshold=focusing_threshold,
            crop_width=crop_indices,
            forward_model_1d=forward_model_N_elements_mask,
            forward_model_2d=forward_model_N_elements_mask_2d,
            zp_init_func=zp_init,
            compute_fzp=False,
        )
        fzp_efficiency = float(fzp_metrics["opt_efficiency"])

        # --- Optimized design with thermal imperfection ---
        opt_x_base_np = np.asarray(opt_x_base)
        opt_profile_full = np.concatenate((opt_x_base_np, np.flip(opt_x_base_np)))
        (
            opt_x_thermal_full,
            opt_thickness_scale_membrane,
            opt_thickness_scale_grating,
        ) = apply_thermal_imperfection(
            opt_profile_full,
            dt_max=t,
            cte_grating=au_cte,
            cte_membrane=sin_cte,
            profile_type="uniform",
        )
        opt_x_thermal_half = opt_x_thermal_full[: opt_x_thermal_full.shape[0] // 2]
        opt_x = torch.tensor(opt_x_thermal_half, device=cuda_device, dtype=torch.float64)

        opt_zs = z_dists
        opt_centers = tuple([(0.0, 0.0) for _ in range(Nelem)])

        opt_elem_params = {
            "thickness": opt_thickness_scale_grating * float(params["element_thickness"]),
            "elem_map": create_material_map("au"),
            "gap_map": [np.array([0, np.inf]), np.array([1.0, 1.0])],
            "membrane_map": create_material_map("si3n4"),
            "membrane_thickness": opt_thickness_scale_membrane * float(params["membrane_thickness"]),
        }

        opt_fwd_model_args = (opt_elem_params, mask, opt_zs, opt_centers)
        opt_metrics = compute_opt_and_fzp_metrics_2d(
            opt_x,
            sim_params_1d,
            opt_fwd_model_args,
            min_feature_size=min_feature_size,
            focusing_threshold=focusing_threshold,
            crop_width=crop_indices,
            forward_model_1d=forward_model_N_elements_mask,
            forward_model_2d=forward_model_N_elements_mask_2d,
            zp_init_func=zp_init,
            compute_fzp=False,
        )
        opt_efficiency = float(opt_metrics["opt_efficiency"])
        opt_width = float(opt_metrics["opt_width"])
        fzp_width = float(fzp_metrics["opt_width"])
        
        fzp_efficiencies[k] = float(fzp_efficiency)
        opt_efficiencies[k] = float(opt_efficiency)
        fzp_widths[k] = float(fzp_width)
        opt_widths[k] = float(opt_width)

    return choice_idx, temp_idx, {
        "fzp_mean": float(np.mean(fzp_efficiencies)),
        "opt_mean": float(np.mean(opt_efficiencies)),
        "fzp_std": float(np.std(fzp_efficiencies)),
        "opt_std": float(np.std(opt_efficiencies)),
        "N_trials": int(N_trials),
        "fzp_width_mean": float(np.mean(fzp_widths)),
        "opt_width_mean": float(np.mean(opt_widths)),
        "fzp_width_std": float(np.std(fzp_widths)),
        "opt_width_std": float(np.std(opt_widths)),
    }


if __name__ == "__main__":
    args = _parse_args()
    start_time = console.script_start(_LOG, argv=sys.argv[1:])
    console.kv(_LOG, "base_id", args.base_id)
    console.kv(_LOG, "run_id", args.run_id)
    console.kv(_LOG, "data_dir", args.data_dir)
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    workers_per_gpu = int(args.workers_per_gpu)
    max_workers = (n_gpus * workers_per_gpu) if n_gpus else workers_per_gpu

    console.runtime_pool(_LOG, n_gpus=n_gpus, workers_per_gpu=workers_per_gpu, max_workers=max_workers)

    results_path, params_path, sweep_arrays_path = resolve_fig1_n_sweep_paths(
        args.data_dir, args.run_id, args.base_id
    )
    results_path = str(results_path)
    params_path = str(params_path)
    sweep_arrays_path = str(sweep_arrays_path)
    console.file_load(_LOG, results_path, label="input results")
    console.file_load(_LOG, params_path, label="input params")
    console.file_load(_LOG, sweep_arrays_path, label="input sweep arrays")

    # Load lightweight sweep arrays in main (worker processes load once via initializer)
    sweep_arrs_main = np.load(sweep_arrays_path, allow_pickle=True).item()
    Nelems = sweep_arrs_main["Nelems"]

    choices = [(0, 1), (0, 2), (0, 3), (0, 5)]
    Nelem_arr = [int(Nelems[ch[1]]) for ch in choices]

    temps = np.linspace(0.0, 450.0, 10)
    console.info(_LOG, f"sweep grid: {len(choices)} cascade choices × {len(temps)} temperatures")

    # output arrays
    fzp_mean_efficiencies = np.zeros((len(choices), len(temps)), dtype=float)
    opt_mean_efficiencies = np.zeros((len(choices), len(temps)), dtype=float)
    fzp_std_efficiencies = np.zeros((len(choices), len(temps)), dtype=float)
    opt_std_efficiencies = np.zeros((len(choices), len(temps)), dtype=float)
    opt_widths = np.zeros((len(choices), len(temps)), dtype=float)
    fzp_widths = np.zeros((len(choices), len(temps)), dtype=float)

    # build tasks
    tasks = []
    for i in range(len(choices)):
        for j in range(len(temps)):
            tasks.append((i, j, choices, temps))

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for w in range(max_workers):
        gpu_id = (w // workers_per_gpu) % n_gpus if n_gpus else 0
        gpu_queue.put(gpu_id)

    pool_kw = dict(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue, results_path, params_path, sweep_arrays_path),
    )

    completed = 0
    total = len(tasks)
    console.info(_LOG, f"submitting {total} robustness tasks to process pool")
    with ProcessPoolExecutor(**pool_kw) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            i, j, out = fut.result()
            fzp_mean_efficiencies[i, j] = out["fzp_mean"]
            opt_mean_efficiencies[i, j] = out["opt_mean"]
            fzp_std_efficiencies[i, j] = out["fzp_std"]
            opt_std_efficiencies[i, j] = out["opt_std"]
            fzp_widths[i, j] = out["fzp_width_mean"]
            opt_widths[i, j] = out["opt_width_mean"]
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                console.progress(
                    _LOG,
                    completed,
                    total,
                    detail=f"last opt_mean={out['opt_mean']:.4f} fzp_mean={out['fzp_mean']:.4f}",
                )

    save_time = get_formatted_datetime()
    save_path = str(
        robustness_results_path(args.data_dir, "fig3d_thermal_robustness_results", args.base_id, save_time)
    )
    np.savez(
        save_path,
        fzp_mean_efficiencies=fzp_mean_efficiencies,
        opt_mean_efficiencies=opt_mean_efficiencies,
        fzp_std_efficiencies=fzp_std_efficiencies,
        opt_std_efficiencies=opt_std_efficiencies,
        fzp_widths=fzp_widths,
        opt_widths=opt_widths,
        choices=np.array(choices, dtype=np.int64),
        temps=temps,
        Nelem_arr=np.array(Nelem_arr, dtype=np.int64),
        ID=np.array([output_id_label(args.base_id)]),
    )
    console.file_saved(_LOG, save_path)
    console.script_done(_LOG, start_time)

