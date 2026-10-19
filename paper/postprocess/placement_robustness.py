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
    width_central_peak,
    get_formatted_datetime,
)
from src.simparams import SimParams  # type: ignore
from src.forwardmodels import forward_model_N_elements_mask_2d  # type: ignore
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

_LOG = "placement_robustness"

DEFAULT_WORKERS_PER_GPU = int(os.environ.get("MAX_WORKERS", "1"))

_worker_gpu_id = None
_opt_rhos = None
_params = None
_sweep_arrs = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate placement robustness from a fig1_N_sweeps run.")
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
    One robustness evaluation for a single (choice_index, error_index).
    Returns CPU scalars only, pickle-safe for ProcessPoolExecutor.
    """
    global _worker_gpu_id
    global _opt_rhos, _params, _sweep_arrs
    choice_idx, error_idx, choices, error_levels = task
    ch = choices[choice_idx]
    console.info(
        _LOG,
        f"worker task choice={ch} error_idx={error_idx} placement_error={float(error_levels[error_idx]):.3e}",
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

    error = float(error_levels[error_idx])

    Nx = int(params["Nx"])
    dx = float(params["dx"])
    f = float(params["f"])
    min_feature_size = float(params["min_feature_size"])
    focusing_threshold = float(params.get("focusing_threshold", 1e-2))

    crop_indices = int(params.get("crop_width", 256))
    N_trials = 5

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

    elem_params = {
        "thickness": float(params["element_thickness"]),
        "elem_map": create_material_map("au"),
        "gap_map": [np.array([0, np.inf]), np.array([1.0, 1.0])],
        "membrane_map": create_material_map("si3n4"),
        "membrane_thickness": float(params["membrane_thickness"]),
    }

    Nelem = int(Nelems[ch[1]])
    z_dists = torch.ones(Nelem - 1, device=cuda_device) * float(params["inter_elem_dist"])
    z_dists = torch.cat((z_dists, torch.tensor([f], device=cuda_device)))
    opt_x = torch.tensor(
        density_half_profile(rhos, ch, int(Nelem * Nx // 2)),
        device=cuda_device,
        dtype=torch.float64,
    )

    fzp_efficiencies = np.zeros((N_trials,), dtype=float)
    opt_efficiencies = np.zeros((N_trials,), dtype=float)

    # precompute circle grid used by both fzp and opt
    y, x = np.ogrid[:crop_indices, :crop_indices]
    dist_sq = (x - crop_indices // 2) ** 2 + (y - crop_indices // 2) ** 2

    for k in range(N_trials):
        # fzp_z = torch.tensor(f + np.random.normal(loc = 0.0, scale = error, size=1))
        fzp_z = f
        fzp_center = tuple(
            [
                (
                    float(np.random.normal(loc=0.0, scale=error, size=1)[0]),
                    float(np.random.normal(loc=0.0, scale=error, size=1)[0]),
                )
            ]
        )
        # fzp_center = tuple([(0.0, 0.0)])

        fzp_x = torch.tensor(
            zp_init(lams[lams.argmax()], f, min_feature_size, 1, sim_params_1d),
            device=cuda_device,
        )
        fzp_fwd_model_args = (elem_params, mask, torch.tensor([fzp_z], device=cuda_device), fzp_center)
        fzp_final_obj, fzp_intensity = forward_model_N_elements_mask_2d(
            fzp_x, sim_params_2d, *fzp_fwd_model_args, inference_only=True, padding=1.0
        )
        fzp_final_intensity = fzp_intensity.detach().cpu().numpy()
        fzp_cropped_intensity = fzp_final_intensity[
            fzp_final_intensity.shape[0] // 2 - crop_indices // 2 : fzp_final_intensity.shape[0] // 2
            + crop_indices // 2,
            fzp_final_intensity.shape[1] // 2 - crop_indices // 2 : fzp_final_intensity.shape[1] // 2
            + crop_indices // 2,
        ]

        fzp_width = width_central_peak(
            fzp_cropped_intensity[fzp_cropped_intensity.shape[0] // 2, :], focusing_threshold
        ) // 2
        if fzp_width > Nx / 10:
            fzp_width = int(2 * 1.22 * min_feature_size / sim_params_1d.dx) // 2

        del fzp_intensity, fzp_final_intensity

        fzp_eff_mask = dist_sq <= fzp_width**2
        fzp_center_pow = fzp_cropped_intensity[fzp_eff_mask].sum()
        fzp_efficiency = fzp_center_pow / (np.pi * (Nx / 2) ** 2)

        # opt_zs = z_dists + torch.normal(mean=0.0, std=error, size=z_dists.shape, device=device)
        # opt_zs = torch.clamp(opt_zs, min = 0.0, max = None)
        opt_centers = tuple(
            [
                (
                    float(np.random.normal(loc=0.0, scale=error, size=1)[0]),
                    float(np.random.normal(loc=0.0, scale=error, size=1)[0]),
                )
                for _ in range(Nelem)
            ]
        )
        opt_zs = z_dists
        # opt_centers = tuple([(0.0, 0.0) for i in range(Nelem)])

        opt_fwd_model_args = (elem_params, mask, opt_zs, opt_centers)
        opt_final_obj, opt_intensity = forward_model_N_elements_mask_2d(
            opt_x, sim_params_2d, *opt_fwd_model_args, inference_only=True, padding=1.0
        )
        opt_final_intensity = opt_intensity.detach().cpu().numpy()
        opt_cropped_intensity = opt_final_intensity[
            opt_final_intensity.shape[0] // 2 - crop_indices // 2 : opt_final_intensity.shape[0] // 2
            + crop_indices // 2,
            opt_final_intensity.shape[1] // 2 - crop_indices // 2 : opt_final_intensity.shape[1] // 2
            + crop_indices // 2,
        ]

        del opt_intensity, opt_final_intensity

        opt_width = width_central_peak(
            opt_cropped_intensity[opt_cropped_intensity.shape[0] // 2, :], focusing_threshold
        ) // 2
        if opt_width > Nx / 10:
            opt_width = int(2 * 1.22 * min_feature_size / sim_params_1d.dx) // 2

        opt_eff_mask = dist_sq <= opt_width**2
        opt_center_pow = opt_cropped_intensity[opt_eff_mask].sum()
        opt_efficiency = opt_center_pow / (np.pi * (Nx / 2) ** 2)

        fzp_efficiencies[k] = float(fzp_efficiency)
        opt_efficiencies[k] = float(opt_efficiency)

    return choice_idx, error_idx, {
        "fzp_mean": float(np.mean(fzp_efficiencies)),
        "opt_mean": float(np.mean(opt_efficiencies)),
        "fzp_std": float(np.std(fzp_efficiencies)),
        "opt_std": float(np.std(opt_efficiencies)),
        "N_trials": int(N_trials),
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

    error_levels = np.logspace(-9, -6, 10)
    placement_precision = error_levels * 1e9
    console.info(_LOG, f"sweep grid: {len(choices)} cascade choices × {len(error_levels)} placement errors")

    # output arrays (same names as original script bottom-of-file)
    fzp_mean_efficiencies = np.zeros((len(choices), len(error_levels)), dtype=float)
    opt_mean_efficiencies = np.zeros((len(choices), len(error_levels)), dtype=float)
    fzp_std_efficiencies = np.zeros((len(choices), len(error_levels)), dtype=float)
    opt_std_efficiencies = np.zeros((len(choices), len(error_levels)), dtype=float)

    # build tasks
    tasks = []
    for i in range(len(choices)):
        for j in range(len(error_levels)):
            tasks.append((i, j, choices, error_levels))

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
        robustness_results_path(
            args.data_dir, "fig3d_placement_robustness_results", args.base_id, save_time
        )
    )
    np.savez(
        save_path,
        fzp_mean_efficiencies=fzp_mean_efficiencies,
        opt_mean_efficiencies=opt_mean_efficiencies,
        fzp_std_efficiencies=fzp_std_efficiencies,
        opt_std_efficiencies=opt_std_efficiencies,
        choices=np.array(choices, dtype=np.int64),
        error_levels=error_levels,
        placement_precision=placement_precision,
        Nelem_arr=np.array(Nelem_arr, dtype=np.int64),
        ID=np.array([output_id_label(args.base_id)]),
    )
    console.file_saved(_LOG, save_path)
    console.script_done(_LOG, start_time)