import copy
import importlib
import itertools
import math
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.sweeps.density_io import (
    INTENSITY_DTYPE,
    compute_fill_factors,
    pack_binary_density,
    save_sweep_results,
)
from src import console
from src.util import get_formatted_datetime


_worker_gpu_id = None


def _init_worker(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()


def _import_module(module_name: str):
    return importlib.import_module(module_name)


def _module_public_dict(module_name: str) -> dict[str, Any]:
    mod = _import_module(module_name)
    out: dict[str, Any] = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        out[name] = getattr(mod, name)
    return out


def _resolve_model(model_key: str, *, dimension: str) -> Callable:
    from src.forwardmodels import (
        forward_model_N_elements_mask,
        forward_model_N_elements_mask_2d,
        forward_model_N_elements_mask_2d_coherent_qdht,
        forward_model_N_elements_mask_partial_coherence,
        forward_model_N_elements_mask_partial_coherence_2d,
        forward_model_N_elements_mask_partial_coherence_2d_qdht,
    )
    if dimension == "opt":
        table = {
            "angular_1d": forward_model_N_elements_mask,
            "qdht_1d": forward_model_N_elements_mask,
            "partial_coherence_1d": forward_model_N_elements_mask_partial_coherence,
        }
    elif dimension == "metric_1d":
        table = {
            "angular_1d": forward_model_N_elements_mask,
            "qdht_1d": forward_model_N_elements_mask,
            "partial_coherence_1d": forward_model_N_elements_mask_partial_coherence,
        }
    else:
        table = {
            "angular_2d": forward_model_N_elements_mask_2d,
            "qdht_2d": forward_model_N_elements_mask_2d,
            "coherent_2d_qdht": forward_model_N_elements_mask_2d_coherent_qdht,
            "partial_coherence_2d": forward_model_N_elements_mask_partial_coherence_2d,
            "partial_coherence_2d_qdht": forward_model_N_elements_mask_partial_coherence_2d_qdht,
        }
    if model_key not in table:
        raise ValueError(f"Unknown model key '{model_key}' for dimension '{dimension}'")
    return table[model_key]


def _method_from_model(model_key: str) -> str:
    # 2D QDHT metric models still call a 2D forward model; `util.compute_opt_and_fzp_metrics_2d`
    # only uses the 1D+spherize path when propagation_method == "qdht" (1D QDHT eval).
    if "2d" in model_key and "qdht" in model_key:
        return "angular"
    return "qdht" if "qdht" in model_key else "angular"


def _build_base_params(param_module: str) -> dict[str, Any]:
    p = _module_public_dict(param_module)
    return {
        "material_default": p["MATERIAL_DEFAULT"],
        "material_map": p["MATERIAL_MAP"],
        "gap_map": p["GAP_MAP_DEFAULT"],
        "membrane_map": p["MEMBRANE_MAP_SI3N4"],
        "N_wvl": p["N_WVL_DEFAULT"],
        "central_energy_ev": p["CENTRAL_ENERGY_EV_DEFAULT"],
        "bandwidth": p["BANDWIDTH_DEFAULT"],
        "Nx": p["NX_DEFAULT"],
        "dx": p["DX_DEFAULT"],
        "min_feature_size": p["MIN_FEATURE_SIZE_DEFAULT"],
        "Nelem": p["N_ELEMENTS_DEFAULT"],
        "f": p["F_DEFAULT"],
        "inter_elem_dist": p["INTER_ELEM_DIST_DEFAULT"],
        "membrane_thickness": p["MEMBRANE_THICKNESS_DEFAULT"],
        "element_thickness": p["ELEMENT_THICKNESS_DEFAULT"],
        "focusing_threshold": p["FOCUSING_THRESHOLD_DEFAULT"],
        "crop_width": p["CROP_WIDTH_DEFAULT"],
        "epsilon": p["EPSILON_DEFAULT"],
        "tolerance": p["TOLERANCE_DEFAULT"],
        "param_tolerance": p["PARAM_TOLERANCE_DEFAULT"],
        "max_eval": p["MAX_EVAL_DEFAULT"],
        "min_beta": p["MIN_BETA_DEFAULT"],
        "beta_schedule": list(p["BETA_SCHEDULE_DEFAULT"]),
        "constraint_fac": p["CONSTRAINT_FAC_DEFAULT"],
        "P": p["P_DEFAULT"],
        "constraint_method": p["CONSTRAINT_METHOD_DEFAULT"],
        "constraint_aggregation": p["CONSTRAINT_AGGREGATION_DEFAULT"],
        "morph_beta": p["MORPH_BETA_DEFAULT"],
        "morph_agg_beta": p["MORPH_AGG_BETA_DEFAULT"],
        "optimization_model": p["OPTIMIZATION_MODEL_DEFAULT"],
        "metric_model_1d": p["METRIC_MODEL_1D_DEFAULT"],
        "metric_model_2d": p["METRIC_MODEL_2D_DEFAULT"],
        "sigma_s": None,
        "sigma_g": None,
        "n_modes": None,
        "center_offsets": None,
    }


def _build_opt_params_template(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "Nelem": int(params["Nelem"]),
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
        "morph_beta": float(params["morph_beta"]),
        "morph_agg_beta": float(params["morph_agg_beta"]),
    }


def _prepare_params_for_save(base_params: dict[str, Any], config: dict[str, Any], axes: dict[str, Any]) -> dict[str, Any]:
    params_for_save = copy.deepcopy(base_params)
    params_for_save.update(config.get("PARAM_OVERRIDES", {}))
    params_for_save["axes"] = {k: np.asarray(v) for k, v in axes.items()}
    for axis_name, axis_values in axes.items():
        params_for_save[axis_name] = np.asarray(axis_values)
    params_for_save["optimizer"] = "run_torch_optimization"
    params_for_save["opt_params_template"] = _build_opt_params_template(params_for_save)
    if "focal_lengths" in axes and "f_default" not in params_for_save:
        params_for_save["f_default"] = float(params_for_save["f"])
    return params_for_save


@dataclass
class SweepRuntimeConfig:
    config_module: str
    max_workers: int | None = None
    workers_per_gpu: int = 2
    n_runs: int | None = None
    save_dir: str | None = None
    dry_run: bool = False


def _default_worker(task: dict[str, Any]) -> dict[str, Any]:
    from src.inversedesign_utils import zp_init
    from src.optimizer import run_torch_optimization
    from src.simparams import SimParams
    from src.util import compute_opt_and_fzp_metrics_2d, focusing_gain, gaussian_energy_spectrum
    global _worker_gpu_id
    gpu_id = _worker_gpu_id if _worker_gpu_id is not None else 0
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
    Ncenter = int(2 * 1.22 * float(params["min_feature_size"]) / dx)
    focusing_mask = torch.zeros(1, Nx, device=device)
    focusing_mask[0, Nx // 2 - Ncenter // 2: Nx // 2 + Ncenter // 2] = 1.0

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

    z_dists = torch.ones(Nelem - 1, device=device) * float(params["inter_elem_dist"])
    z_dists = torch.cat((z_dists, torch.tensor([float(params["f"])], device=device)))
    fwd_model_args = (elem_params, focusing_mask, z_dists, params.get("center_offsets"))

    opt_params = {
        "Nelem": Nelem,
        "min_feature_size": float(params["min_feature_size"]),
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
        fwd_model_args,
        objective_function=objective_function,
    )
    x_tensor = torch.tensor(raw_design, dtype=torch.float64, device=device)
    rho_tilde, _ = model.filter_density(x_tensor)
    rho_bar = (rho_tilde > 0.5).to(dtype=float)

    metrics = compute_opt_and_fzp_metrics_2d(
        rho_bar,
        sim_params,
        fwd_model_args,
        min_feature_size=float(params["min_feature_size"]),
        focusing_threshold=float(params["focusing_threshold"]),
        crop_width=int(params["crop_width"]),
        forward_model_1d=metric_1d,
        forward_model_2d=metric_2d,
        zp_init_func=zp_init,
    )
    opt_gain = focusing_gain(np.asarray(metrics["opt_intensity_1d"]), float(params["focusing_threshold"]))
    fzp_gain = focusing_gain(np.asarray(metrics["fzp_intensity_1d"]), float(params["focusing_threshold"]))
    obj_np = np.array([float(o) if hasattr(o, "item") else float(o) for o in obj_list], dtype=np.float64)
    console.info(
        "sweep.worker",
        (
            f"task {task_id} finished: opt_eff={float(metrics['opt_efficiency']):.4f} "
            f"fzp_eff={float(metrics['fzp_efficiency']):.4f} "
            f"opt_gain={opt_gain:.4f} fzp_gain={fzp_gain:.4f}"
        ),
    )
    return {
        "status": "ok",
        "index": tuple(task["index"]),
        "task_id": int(task["task_id"]),
        "result": {
            "rho_bar": pack_binary_density(rho_bar.detach().cpu().numpy()),
            "obj_list": obj_np,
            "opt_obj": float(metrics["opt_final_obj"]),
            "opt_eff": float(metrics["opt_efficiency"]),
            "opt_width": float(metrics["opt_width"]),
            "opt_gain": float(opt_gain),
            "opt_intensity": np.asarray(metrics["opt_intensity_1d"], dtype=INTENSITY_DTYPE),
            "fzp_obj": float(metrics["fzp_final_obj"]),
            "fzp_eff": float(metrics["fzp_efficiency"]),
            "fzp_width": float(metrics["fzp_width"]),
            "fzp_gain": float(fzp_gain),
            "fzp_intensity": np.asarray(metrics["fzp_intensity_1d"], dtype=INTENSITY_DTYPE),
        },
    }


def _build_tasks(config_module: str, base_params: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = _module_public_dict(config_module)
    axes: dict[str, Any] = dict(config["SWEEP_AXES"])
    axis_names = list(axes.keys())
    axis_arrays = [np.asarray(axes[k]) for k in axis_names]
    index_product = list(itertools.product(*[range(len(a)) for a in axis_arrays]))
    param_overrides = dict(config.get("PARAM_OVERRIDES", {}))
    task_cost_fn = config.get("task_cost_fn")
    build_point_overrides = config["build_point_overrides"]
    tasks = []
    for point_index in index_product:
        axis_values = {k: axis_arrays[ii][point_index[ii]] for ii, k in enumerate(axis_names)}
        params = copy.deepcopy(base_params)
        params.update(param_overrides)
        overrides = dict(build_point_overrides(point_index, axis_values, params))
        params.update(overrides)
        params.setdefault("material_map", base_params["material_map"])
        if "material" in params and "material_map" not in overrides:
            from src.util import create_material_map
            params["material_map"] = create_material_map(str(params["material"]))
        cost = 1.0
        if callable(task_cost_fn):
            cost = float(task_cost_fn(point_index, axis_values, params))
        tasks.append({"index": point_index, "axis_values": axis_values, "params": params, "cost": cost})
    tasks_sorted = sorted(tasks, key=lambda t: t["cost"])
    half = (len(tasks_sorted) + 1) // 2
    light = tasks_sorted[:half]
    heavy = tasks_sorted[half:]
    interleaved = []
    for i in range(max(len(light), len(heavy))):
        if i < len(light):
            interleaved.append(light[i])
        if i < len(heavy):
            interleaved.append(heavy[i])
    for task_id, task in enumerate(interleaved, start=1):
        task["task_id"] = task_id
    return interleaved, config, axes


def _collect_results(results: list[dict[str, Any]], axes: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(len(np.asarray(axes[k])) for k in axes.keys())
    max_params = int(config.get("MAX_PARAMS", 0))
    nx_store = int(config.get("NX_STORE", 0))
    out = {
        "opt_rhos": np.zeros(
            shape + ((max_params,) if max_params > 0 else (1,)),
            dtype=np.bool_,
        ),
        "obj_lists": np.empty(shape, dtype=object),
        "opt_objs": np.full(shape, np.nan),
        "opt_efficiencies": np.full(shape, np.nan),
        "opt_gains": np.full(shape, np.nan),
        "opt_widths": np.full(shape, np.nan),
        "fzp_objs": np.full(shape, np.nan),
        "fzp_efficiencies": np.full(shape, np.nan),
        "fzp_gains": np.full(shape, np.nan),
        "fzp_widths": np.full(shape, np.nan),
    }
    if nx_store > 0:
        out["opt_intensities"] = np.zeros(shape + (nx_store,), dtype=INTENSITY_DTYPE)
        out["fzp_intensities"] = np.zeros(shape + (nx_store,), dtype=INTENSITY_DTYPE)
    failed = []
    for item in results:
        idx = tuple(item["index"])
        if item["status"] != "ok":
            failed.append(item)
            continue
        r = item["result"]
        rho = pack_binary_density(r["rho_bar"])
        if max_params > 0 and rho.shape[0] < max_params:
            rho = np.pad(rho, (0, max_params - int(rho.shape[0])), mode="constant", constant_values=False)
        out["opt_rhos"][idx] = rho if max_params > 0 else rho[:1]
        out["obj_lists"][idx] = r["obj_list"]
        out["opt_objs"][idx] = r["opt_obj"]
        out["opt_efficiencies"][idx] = r["opt_eff"]
        out["opt_gains"][idx] = r["opt_gain"]
        out["opt_widths"][idx] = r["opt_width"]
        out["fzp_objs"][idx] = r["fzp_obj"]
        out["fzp_efficiencies"][idx] = r["fzp_eff"]
        out["fzp_gains"][idx] = r["fzp_gain"]
        out["fzp_widths"][idx] = r["fzp_width"]
        if nx_store > 0:
            oi = np.asarray(r["opt_intensity"])
            fi = np.asarray(r["fzp_intensity"])
            if oi.shape[0] < nx_store:
                oi = np.pad(oi, (0, nx_store - int(oi.shape[0])), mode="constant")
            if fi.shape[0] < nx_store:
                fi = np.pad(fi, (0, nx_store - int(fi.shape[0])), mode="constant")
            out["opt_intensities"][idx] = np.asarray(oi, dtype=INTENSITY_DTYPE)
            out["fzp_intensities"][idx] = np.asarray(fi, dtype=INTENSITY_DTYPE)
    out["failed_task_ids"] = np.asarray([f["task_id"] for f in failed], dtype=np.int64)
    out["failed_task_indices"] = np.asarray([f["index"] for f in failed], dtype=np.int64).reshape(-1, len(shape))
    out["failed_task_errors"] = np.asarray(
        [f"{f.get('error_type','Error')}: {f.get('error_message', 'unknown')}" for f in failed],
        dtype=object,
    )
    nelem_axis_name = next((name for name in ("Nelems", "Nelem") if name in axes), None)
    if nelem_axis_name is not None and nx_store > 0:
        nelem_axis = list(axes.keys()).index(nelem_axis_name)
        out["fill_factors"] = compute_fill_factors(
            out["opt_rhos"],
            axes[nelem_axis_name],
            nx=nx_store,
            nelem_axis=nelem_axis,
        )
    return out


def run_sweep(runtime: SweepRuntimeConfig) -> None:
    log = "sweep"
    sweep_start = console.script_start(log, argv=[f"config={runtime.config_module}"])
    base = _build_base_params("paper.sweeps.standard_params")
    tasks, config, axes = _build_tasks(runtime.config_module, base)
    save_prefix = str(config["SAVE_PREFIX"])
    save_dir = runtime.save_dir or config.get("SAVE_DIR") or os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "outputs")
    n_runs = int(runtime.n_runs if runtime.n_runs is not None else config.get("N_RUNS", 1))
    save_run_suffix = bool(config.get("SAVE_RUN_SUFFIX", True)) or n_runs > 1
    console.kv(log, "save_prefix", save_prefix)
    console.kv(log, "save_dir", save_dir)
    console.kv(log, "n_tasks", len(tasks))
    console.kv(log, "n_runs", n_runs)
    console.describe_axes(log, axes)
    if runtime.dry_run:
        console.info(log, f"dry-run only (no simulations); config={runtime.config_module}")
        console.script_done(log, sweep_start)
        return
    os.makedirs(save_dir, exist_ok=True)
    console.info(log, f"created output directory {save_dir}")

    n_gpus = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    workers_per_gpu = int(runtime.workers_per_gpu)
    max_workers = int(runtime.max_workers) if runtime.max_workers is not None else ((n_gpus * workers_per_gpu) if n_gpus else workers_per_gpu)
    console.runtime_pool(log, n_gpus=n_gpus, workers_per_gpu=workers_per_gpu, max_workers=max_workers)
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    console.info(log, "set OMP/MKL/OpenBLAS threads to 1 for worker processes")

    save_time = get_formatted_datetime()
    params_for_save = _prepare_params_for_save(base, config, axes)
    params_path = f"{save_dir}/{save_prefix}_params_{save_time}.npy"
    arrays_path = f"{save_dir}/{save_prefix}_sweep_arrays_{save_time}.npy"
    np.save(params_path, params_for_save)
    np.save(arrays_path, {k: np.asarray(v) for k, v in axes.items()})
    console.file_saved(log, params_path)
    console.file_saved(log, arrays_path)

    worker_fn = config.get("worker_fn", _default_worker)
    for run_id in range(n_runs):
        console.banner(log, f"run {run_id + 1}/{n_runs}")
        run_tasks = []
        for t in tasks:
            td = dict(t)
            td["task_id"] = int(t["task_id"])
            run_tasks.append(td)
        for i in range(max_workers):
            gpu_id = (i // workers_per_gpu) % n_gpus if n_gpus else 0
            gpu_queue.put(gpu_id)
        start = time.time()
        console.info(log, f"submitting {len(run_tasks)} tasks to process pool")
        results: list[dict[str, Any]] = []
        completed = 0
        total = len(run_tasks)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(gpu_queue,),
        ) as ex:
            futures = [ex.submit(worker_fn, t) for t in run_tasks]
            for fut in as_completed(futures):
                try:
                    item = fut.result()
                    results.append(item)
                    completed += 1
                    status = item.get("status", "ok")
                    task_id = item.get("task_id", -1)
                    if status == "ok":
                        r = item.get("result", {})
                        console.info(
                            log,
                            (
                                f"task {task_id} collected: opt_eff={float(r.get('opt_eff', float('nan'))):.4f} "
                                f"fzp_eff={float(r.get('fzp_eff', float('nan'))):.4f}"
                            ),
                        )
                    else:
                        console.warn(
                            log,
                            f"task {task_id} failed: {item.get('error_type')}: {item.get('error_message')}",
                        )
                    if completed % max(1, total // 20) == 0 or completed == total:
                        console.progress(log, completed, total)
                except BaseException as exc:
                    completed += 1
                    console.error(log, f"worker raised {type(exc).__name__}: {exc}")
                    results.append(
                        {
                            "status": "error",
                            "task_id": -1,
                            "index": tuple(),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if completed % max(1, total // 20) == 0 or completed == total:
                        console.progress(log, completed, total)
        arrays = _collect_results(results, axes, config)
        n_failed = len(arrays.get("failed_task_ids", []))
        if n_failed:
            console.warn(log, f"{n_failed} task(s) failed; see failed_task_* arrays in output")
        if save_run_suffix:
            arrays["run_id"] = np.int64(run_id)
            result_path = f"{save_dir}/{save_prefix}_results_{save_time}_run_{run_id}.npz"
        else:
            result_path = f"{save_dir}/{save_prefix}_results_{save_time}.npz"
        save_sweep_results(result_path, arrays, keys=config.get("RESULTS_SAVE_KEYS"))
        console.file_saved(log, result_path)
        console.elapsed(log, f"run {run_id + 1}/{n_runs} for {save_prefix}", time.time() - start)
    console.script_done(log, sweep_start)
