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
    apply_morphological_error_1d,
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

_LOG = "erosion_dilation_robustness"

DEFAULT_WORKERS_PER_GPU = int(os.environ.get("MAX_WORKERS", "4"))


# Edit this array (length units matching `params["dx"]`) to choose the erosion/dilation magnitudes.
# Each level becomes a continuous pixel radius via: strength_pixels = abs(level) / dx
# (see `apply_morphological_error_1d` in `src/util.py`).
morph_error_levels = np.linspace(0, 50e-9, 10)

# Which operations to sweep (must match `apply_morphological_error_1d` naming).
morph_operations = ("erode", "dilate")


_worker_gpu_id = None
_opt_rhos = None
_params = None
_sweep_arrs = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate erosion/dilation robustness from a fig1_N_sweeps run.")
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


def _apply_morphology_to_cascade_vector(
    x_np: np.ndarray,
    *,
    Nelem: int,
    seg_len: int,
    operation: str,
    strength_pixels: float,
) -> np.ndarray:
    """
    Apply morphological erosion/dilation independently to each element's 1D radial profile segment.

    `forward_model_N_elements_mask_2d` splits x into `Nelem` equal parts, each part is a radial profile
    (center-to-edge) of length `seg_len = Nx//2`. This function preserves that convention.
    """
    out = np.empty_like(x_np)
    for k in range(Nelem):
        seg = x_np[k * seg_len : (k + 1) * seg_len]
        out[k * seg_len : (k + 1) * seg_len] = apply_morphological_error_1d(
            seg,
            operation=operation,
            strength=strength_pixels,
        )
    return out


def worker(task):
    """
    One robustness evaluation for a single (choice_index, level_index, operation_index).

    Returns CPU scalars only, pickle-safe for ProcessPoolExecutor.
    """
    global _worker_gpu_id
    global _opt_rhos, _params, _sweep_arrs

    choice_idx, level_idx, op_idx, choices, levels = task
    ch = choices[choice_idx]
    operation = morph_operations[op_idx]
    console.info(
        _LOG,
        (
            f"worker task choice={ch} level_idx={level_idx} op={operation} "
            f"morph_level={float(levels[level_idx]):.3e} m"
        ),
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

    level = float(levels[level_idx])

    Nx = int(params["Nx"])
    dx = float(params["dx"])
    f = float(params["f"])
    min_feature_size = float(params["min_feature_size"])
    focusing_threshold = float(params.get("focusing_threshold", 1e-2))

    crop_indices = int(params.get("crop_width", 256))
    N_trials = 1

    strength_pixels = float(abs(level) / dx)
    if strength_pixels < 0:
        strength_pixels = 0.0

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
    seg_len = Nx // 2

    # Convert once to numpy so morphological ops are purely CPU/scipy.
    opt_x_base_np = density_half_profile(rhos, ch, int(Nelem * seg_len))

    fzp_mean_efficiencies = np.zeros((N_trials,), dtype=float)
    opt_mean_efficiencies = np.zeros((N_trials,), dtype=float)

    # precompute circle grid used by both fzp and opt
    y, x = np.ogrid[:crop_indices, :crop_indices]
    dist_sq = (x - crop_indices // 2) ** 2 + (y - crop_indices // 2) ** 2

    for k in range(N_trials):
        # --- FZP (single element @ f) ---
        fzp_x_np = zp_init(lams[lams.argmax()], f, min_feature_size, 1, sim_params_1d)
        fzp_x_np_err = apply_morphological_error_1d(
            np.asarray(fzp_x_np, dtype=np.float64),
            operation=operation,
            strength=strength_pixels,
        )
        fzp_x = torch.tensor(fzp_x_np_err, device=cuda_device, dtype=torch.float64)

        fzp_z = torch.tensor([f], device=cuda_device)
        fzp_center = ((0.0, 0.0),)

        fzp_fwd_model_args = (elem_params, mask, fzp_z, fzp_center)
        _, fzp_intensity = forward_model_N_elements_mask_2d(
            fzp_x,
            sim_params_2d,
            *fzp_fwd_model_args,
            inference_only=True,
            padding=1.0,
        )
        fzp_intensity_np = fzp_intensity.detach().cpu().numpy()

        fzp_cropped_intensity = fzp_intensity_np[
            fzp_intensity_np.shape[0] // 2 - crop_indices // 2 : fzp_intensity_np.shape[0] // 2
            + crop_indices // 2,
            fzp_intensity_np.shape[1] // 2 - crop_indices // 2 : fzp_intensity_np.shape[1] // 2
            + crop_indices // 2,
        ]

        fzp_width = (
            width_central_peak(fzp_cropped_intensity[fzp_cropped_intensity.shape[0] // 2, :], focusing_threshold)
            // 2
        )
        if fzp_width > Nx / 10:
            fzp_width = int(2 * 1.22 * min_feature_size / sim_params_1d.dx) // 2

        fzp_eff_mask = dist_sq <= fzp_width**2
        fzp_center_pow = fzp_cropped_intensity[fzp_eff_mask].sum()
        fzp_efficiency = fzp_center_pow / (np.pi * (Nx / 2) ** 2)

        # --- Optimized multi-element design ---
        opt_x_np_err = _apply_morphology_to_cascade_vector(
            opt_x_base_np,
            Nelem=Nelem,
            seg_len=seg_len,
            operation=operation,
            strength_pixels=strength_pixels,
        )
        opt_x = torch.tensor(opt_x_np_err, device=cuda_device, dtype=torch.float64)

        opt_centers = tuple((0.0, 0.0) for _ in range(Nelem))
        opt_fwd_model_args = (elem_params, mask, z_dists, opt_centers)

        _, opt_intensity = forward_model_N_elements_mask_2d(
            opt_x,
            sim_params_2d,
            *opt_fwd_model_args,
            inference_only=True,
            padding=1.0,
        )
        opt_intensity_np = opt_intensity.detach().cpu().numpy()

        opt_cropped_intensity = opt_intensity_np[
            opt_intensity_np.shape[0] // 2 - crop_indices // 2 : opt_intensity_np.shape[0] // 2
            + crop_indices // 2,
            opt_intensity_np.shape[1] // 2 - crop_indices // 2 : opt_intensity_np.shape[1] // 2
            + crop_indices // 2,
        ]

        opt_width = (
            width_central_peak(opt_cropped_intensity[opt_cropped_intensity.shape[0] // 2, :], focusing_threshold)
            // 2
        )
        if opt_width > Nx / 10:
            opt_width = int(2 * 1.22 * min_feature_size / sim_params_1d.dx) // 2

        opt_eff_mask = dist_sq <= opt_width**2
        opt_center_pow = opt_cropped_intensity[opt_eff_mask].sum()
        opt_efficiency = opt_center_pow / (np.pi * (Nx / 2) ** 2)

        fzp_mean_efficiencies[k] = float(fzp_efficiency)
        opt_mean_efficiencies[k] = float(opt_efficiency)

        # Help memory a bit in long workers.
        del fzp_intensity_np, fzp_cropped_intensity
        del opt_intensity_np, opt_cropped_intensity
        del fzp_x, opt_x

    return choice_idx, level_idx, op_idx, {
        "fzp_mean": float(np.mean(fzp_mean_efficiencies)),
        "opt_mean": float(np.mean(opt_mean_efficiencies)),
        "fzp_std": float(np.std(fzp_mean_efficiencies)),
        "opt_std": float(np.std(opt_mean_efficiencies)),
        "N_trials": int(N_trials),
        "strength_pixels": float(strength_pixels),
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

    # Load dx in main so we can store the pixel-iteration strengths alongside results.
    params_main = np.load(params_path, allow_pickle=True).item()
    dx_main = float(params_main["dx"])

    choices = [(0, 1), (0, 2), (0, 3), (0, 5)]
    Nelem_arr = [int(Nelems[ch[1]]) for ch in choices]

    levels = np.asarray(morph_error_levels, dtype=float)

    strength_pixels_arr = np.array([float(abs(l) / dx_main) for l in levels], dtype=np.float64)

    n_choices = len(choices)
    n_levels = len(levels)
    n_ops = len(morph_operations)
    console.info(
        _LOG,
        f"sweep grid: {n_choices} cascade choices × {n_levels} morph levels × {n_ops} operations",
    )

    # output arrays
    fzp_mean_efficiencies = np.zeros((n_choices, n_levels, n_ops), dtype=float)
    opt_mean_efficiencies = np.zeros((n_choices, n_levels, n_ops), dtype=float)
    fzp_std_efficiencies = np.zeros((n_choices, n_levels, n_ops), dtype=float)
    opt_std_efficiencies = np.zeros((n_choices, n_levels, n_ops), dtype=float)

    # build tasks
    tasks = []
    for i in range(n_choices):
        for j in range(n_levels):
            for k in range(n_ops):
                tasks.append((i, j, k, choices, levels))

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
            choice_idx, level_idx, op_idx, out = fut.result()
            fzp_mean_efficiencies[choice_idx, level_idx, op_idx] = out["fzp_mean"]
            opt_mean_efficiencies[choice_idx, level_idx, op_idx] = out["opt_mean"]
            fzp_std_efficiencies[choice_idx, level_idx, op_idx] = out["fzp_std"]
            opt_std_efficiencies[choice_idx, level_idx, op_idx] = out["opt_std"]

            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                console.progress(
                    _LOG,
                    completed,
                    total,
                    detail=f"last opt_mean={out['opt_mean']:.4f} fzp_mean={out['fzp_mean']:.4f}",
                )

    save_time = get_formatted_datetime()
    output_prefixes = {
        "erode": "fig3d_erosion_robustness_results",
        "dilate": "figS4a_dilation_robustness_results",
    }
    for op_idx, operation in enumerate(morph_operations):
        save_path = str(
            robustness_results_path(
                args.data_dir, output_prefixes[operation], args.base_id, save_time
            )
        )
        np.savez(
            save_path,
            fzp_mean_efficiencies=fzp_mean_efficiencies[:, :, op_idx],
            opt_mean_efficiencies=opt_mean_efficiencies[:, :, op_idx],
            fzp_std_efficiencies=fzp_std_efficiencies[:, :, op_idx],
            opt_std_efficiencies=opt_std_efficiencies[:, :, op_idx],
            choices=np.array(choices, dtype=np.int64),
            Nelem_arr=np.array(Nelem_arr, dtype=np.int64),
            morph_error_levels=levels,
            morph_operations=np.array([operation], dtype=np.str_),
            strength_pixels_arr=strength_pixels_arr,
            ID=np.array([output_id_label(args.base_id)]),
        )
        console.file_saved(_LOG, save_path)
    console.script_done(_LOG, start_time)
