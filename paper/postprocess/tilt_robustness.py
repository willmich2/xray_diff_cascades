import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.data_paths import (  # type: ignore
    DEFAULT_BASE_SWEEP_ID,
    DEFAULT_DATA_DIR,
    output_id_label,
    resolve_fig1_n_sweep_paths,
    robustness_results_path,
)
from paper.sweeps.density_io import density_half_profile, load_opt_rhos  # type: ignore
from src import console  # type: ignore
from src.forwardmodels import forward_model_N_elements_mask_2d  # type: ignore
from src.inversedesign_utils import zp_init  # type: ignore
from src.simparams import SimParams  # type: ignore
from src.util import (  # type: ignore
    create_material_map,
    gaussian_energy_spectrum,
    get_formatted_datetime,
    width_central_peak,
)

_LOG = "tilt_robustness"

DEFAULT_WORKERS_PER_GPU = int(os.environ.get("MAX_WORKERS", "1"))

_worker_gpu_id = None
_opt_rhos = None
_params = None
_sweep_arrs = None
_n_trials = 5
_seed = 12345


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate element-tilt robustness from a fig1_N_sweeps run."
    )
    parser.add_argument(
        "--base-id",
        default=DEFAULT_BASE_SWEEP_ID,
        help="Timestamp for fig1_N_sweeps_* inputs; omit for bundled paper_data/.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run index for the selected fig1_N_sweeps results file.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing sweep/result files.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def _init_worker(
    gpu_queue,
    results_path: str,
    params_path: str,
    sweep_arrays_path: str,
    n_trials: int,
    seed: int,
):
    global _worker_gpu_id, _opt_rhos, _params, _sweep_arrs, _n_trials, _seed
    _worker_gpu_id = gpu_queue.get()
    with np.load(results_path, allow_pickle=True) as results_np:
        _opt_rhos = load_opt_rhos(results_np["opt_rhos"])
    _params = np.load(params_path, allow_pickle=True).item()
    _sweep_arrs = np.load(sweep_arrays_path, allow_pickle=True).item()
    _n_trials = int(n_trials)
    _seed = int(seed)


def _focusing_efficiency(
    intensity: torch.Tensor,
    *,
    crop_indices: int,
    Nx: int,
    min_feature_size: float,
    dx: float,
    focusing_threshold: float,
) -> float:
    intensity_np = intensity.detach().cpu().numpy()
    cy, cx = intensity_np.shape[0] // 2, intensity_np.shape[1] // 2
    half = crop_indices // 2
    cropped = intensity_np[cy - half : cy + half, cx - half : cx + half]

    width = width_central_peak(cropped[cropped.shape[0] // 2, :], focusing_threshold) // 2
    if width > Nx / 10:
        width = int(2 * 1.22 * min_feature_size / dx) // 2

    y, x = np.ogrid[: cropped.shape[0], : cropped.shape[1]]
    dist_sq = (x - cropped.shape[1] // 2) ** 2 + (y - cropped.shape[0] // 2) ** 2
    center_power = cropped[dist_sq <= width**2].sum()
    return float(center_power / (np.pi * (Nx / 2) ** 2))


def worker(task):
    """
    Evaluate one cascade choice and one per-axis tilt standard deviation.

    Every element receives independent x/y rotations drawn from N(0, error^2).
    Rotations are in radians and are applied about the element center.
    """
    global _worker_gpu_id, _opt_rhos, _params, _sweep_arrs, _n_trials, _seed
    choice_idx, error_idx, choices, error_levels = task
    choice = choices[choice_idx]
    error = float(error_levels[error_idx])
    console.info(
        _LOG,
        f"worker task choice={choice} error_idx={error_idx} tilt_std={error:.3e} rad",
    )

    gpu_id = _worker_gpu_id if _worker_gpu_id is not None else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = (
        torch.device("cuda", gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if _opt_rhos is None or _params is None or _sweep_arrs is None:
        raise RuntimeError("Worker globals not initialized. Did initializer run?")

    params = _params
    Nelems = _sweep_arrs["Nelems"]
    Nelem = int(Nelems[choice[1]])
    Nx = int(params["Nx"])
    dx = float(params["dx"])
    f = float(params["f"])
    min_feature_size = float(params["min_feature_size"])
    focusing_threshold = float(params.get("focusing_threshold", 1e-2))
    crop_indices = int(params.get("crop_width", 256))

    lams, weights = gaussian_energy_spectrum(
        central_energy_ev=float(params["central_energy_ev"]),
        N=int(params["N_wvl"]),
        bandwidth=float(params["bandwidth"]),
        device=device,
        bandwidth_in_wavelength=False,
    )
    sim_params_1d = SimParams(
        Ny=1,
        Nx=Nx,
        dx=dx,
        device=device,
        dtype=torch.complex128,
        lams=lams,
        weights=weights,
    )
    sim_params_2d = SimParams(
        Ny=Nx,
        Nx=Nx,
        dx=dx,
        device=device,
        dtype=torch.complex128,
        lams=lams,
        weights=weights,
    )

    Ncenter = int(2 * 1.22 * min_feature_size / dx)
    mask = torch.zeros(1, Nx, device=device)
    mask[0, Nx // 2 - Ncenter // 2 : Nx // 2 + Ncenter // 2] = 1.0
    elem_params = {
        "thickness": float(params["element_thickness"]),
        "elem_map": create_material_map("au"),
        "gap_map": [np.array([0, np.inf]), np.array([1.0, 1.0])],
        "membrane_map": create_material_map("si3n4"),
        "membrane_thickness": float(params["membrane_thickness"]),
    }

    z_dists = torch.ones(Nelem - 1, device=device) * float(params["inter_elem_dist"])
    z_dists = torch.cat((z_dists, torch.tensor([f], device=device)))
    opt_x = torch.tensor(
        density_half_profile(_opt_rhos, choice, int(Nelem * Nx // 2)),
        device=device,
        dtype=torch.float64,
    )
    fzp_x = torch.tensor(
        zp_init(lams[lams.argmax()], f, min_feature_size, 1, sim_params_1d),
        device=device,
        dtype=torch.float64,
    )

    fzp_efficiencies = np.zeros(_n_trials, dtype=float)
    opt_efficiencies = np.zeros(_n_trials, dtype=float)
    # The FZP seed is independent of cascade choice so its saved curve is
    # identical across the choice axis, matching its role as one baseline.
    fzp_rng = np.random.default_rng(_seed + 1009 * error_idx)
    opt_rng = np.random.default_rng(_seed + 1009 * error_idx + 104729 * choice_idx)
    zero_fzp_centers = ((0.0, 0.0),)
    zero_opt_centers = tuple((0.0, 0.0) for _ in range(Nelem))

    for trial_idx in range(_n_trials):
        fzp_tilts = tuple(
            tuple(float(v) for v in fzp_rng.normal(0.0, error, size=2))
            for _ in range(1)
        )
        opt_tilts = tuple(
            tuple(float(v) for v in opt_rng.normal(0.0, error, size=2))
            for _ in range(Nelem)
        )

        _, fzp_intensity = forward_model_N_elements_mask_2d(
            fzp_x,
            sim_params_2d,
            elem_params,
            mask,
            torch.tensor([f], device=device),
            zero_fzp_centers,
            inference_only=True,
            padding=1.0,
            tilt_angles=fzp_tilts,
        )
        fzp_efficiencies[trial_idx] = _focusing_efficiency(
            fzp_intensity,
            crop_indices=crop_indices,
            Nx=Nx,
            min_feature_size=min_feature_size,
            dx=dx,
            focusing_threshold=focusing_threshold,
        )
        del fzp_intensity

        _, opt_intensity = forward_model_N_elements_mask_2d(
            opt_x,
            sim_params_2d,
            elem_params,
            mask,
            z_dists,
            zero_opt_centers,
            inference_only=True,
            padding=1.0,
            tilt_angles=opt_tilts,
        )
        opt_efficiencies[trial_idx] = _focusing_efficiency(
            opt_intensity,
            crop_indices=crop_indices,
            Nx=Nx,
            min_feature_size=min_feature_size,
            dx=dx,
            focusing_threshold=focusing_threshold,
        )
        del opt_intensity

    return choice_idx, error_idx, {
        "fzp_mean": float(np.mean(fzp_efficiencies)),
        "opt_mean": float(np.mean(opt_efficiencies)),
        "fzp_std": float(np.std(fzp_efficiencies)),
        "opt_std": float(np.std(opt_efficiencies)),
    }


def main() -> None:
    args = _parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    start_time = console.script_start(_LOG, argv=sys.argv[1:])
    console.kv(_LOG, "base_id", args.base_id)
    console.kv(_LOG, "run_id", args.run_id)
    console.kv(_LOG, "data_dir", args.data_dir)
    console.kv(_LOG, "trials", args.trials)
    console.kv(_LOG, "seed", args.seed)
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    workers_per_gpu = int(args.workers_per_gpu)
    max_workers = n_gpus * workers_per_gpu if n_gpus else workers_per_gpu
    console.runtime_pool(
        _LOG,
        n_gpus=n_gpus,
        workers_per_gpu=workers_per_gpu,
        max_workers=max_workers,
    )

    results_path, params_path, sweep_arrays_path = resolve_fig1_n_sweep_paths(
        args.data_dir, args.run_id, args.base_id
    )
    console.file_load(_LOG, str(results_path), label="input results")
    console.file_load(_LOG, str(params_path), label="input params")
    console.file_load(_LOG, str(sweep_arrays_path), label="input sweep arrays")

    sweep_arrs_main = np.load(sweep_arrays_path, allow_pickle=True).item()
    Nelems = sweep_arrs_main["Nelems"]
    choices = [(0, 1), (0, 2), (0, 3), (0, 5)]
    Nelem_arr = [int(Nelems[choice[1]]) for choice in choices]

    error_levels = np.logspace(-4, -0.5, 10)
    tilt_error_mrad = error_levels * 1e3
    console.info(
        _LOG,
        f"sweep grid: {len(choices)} cascade choices × {len(error_levels)} tilt errors",
    )

    shape = (len(choices), len(error_levels))
    fzp_mean_efficiencies = np.zeros(shape, dtype=float)
    opt_mean_efficiencies = np.zeros(shape, dtype=float)
    fzp_std_efficiencies = np.zeros(shape, dtype=float)
    opt_std_efficiencies = np.zeros(shape, dtype=float)
    tasks = [
        (i, j, choices, error_levels)
        for i in range(len(choices))
        for j in range(len(error_levels))
    ]

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for worker_idx in range(max_workers):
        gpu_id = (worker_idx // workers_per_gpu) % n_gpus if n_gpus else 0
        gpu_queue.put(gpu_id)

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            gpu_queue,
            str(results_path),
            str(params_path),
            str(sweep_arrays_path),
            args.trials,
            args.seed,
        ),
    ) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            i, j, out = future.result()
            fzp_mean_efficiencies[i, j] = out["fzp_mean"]
            opt_mean_efficiencies[i, j] = out["opt_mean"]
            fzp_std_efficiencies[i, j] = out["fzp_std"]
            opt_std_efficiencies[i, j] = out["opt_std"]
            completed += 1
            console.progress(
                _LOG,
                completed,
                len(tasks),
                detail=f"last opt_mean={out['opt_mean']:.4f} fzp_mean={out['fzp_mean']:.4f}",
            )

    save_time = get_formatted_datetime()
    save_path = robustness_results_path(
        args.data_dir,
        "figS4c_tilt_robustness_results",
        args.base_id,
        save_time,
    )
    np.savez(
        save_path,
        fzp_mean_efficiencies=fzp_mean_efficiencies,
        opt_mean_efficiencies=opt_mean_efficiencies,
        fzp_std_efficiencies=fzp_std_efficiencies,
        opt_std_efficiencies=opt_std_efficiencies,
        choices=np.array(choices, dtype=np.int64),
        error_levels=error_levels,
        tilt_error_mrad=tilt_error_mrad,
        Nelem_arr=np.array(Nelem_arr, dtype=np.int64),
        N_trials=np.array([args.trials], dtype=np.int64),
        seed=np.array([args.seed], dtype=np.int64),
        tilt_distribution=np.array(["independent per-axis zero-mean Gaussian"]),
        ID=np.array([output_id_label(args.base_id)]),
    )
    console.file_saved(_LOG, str(save_path))
    console.script_done(_LOG, start_time)


if __name__ == "__main__":
    main()
