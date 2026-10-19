import math

import numpy as np

from paper.sweeps.standard_params import N_ELEMENTS_DEFAULT, NX_DEFAULT, DX_DEFAULT

SAVE_PREFIX = "figS1_coherence_illumination_sweep"
SAVE_DIR = None
N_RUNS = 1
_aperture_width = NX_DEFAULT * DX_DEFAULT
SWEEP_AXES = {
    "sigma_g": _aperture_width * np.logspace(-1, 1.5, 15),
    "sigma_s": _aperture_width * np.logspace(-1, 1.5, 15),
}
TAIL_TOLERANCE = 1e-3
MAX_MODES = 50
MAX_PARAMS = int(N_ELEMENTS_DEFAULT * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)
PARAM_OVERRIDES = {
    "optimization_model": "partial_coherence_1d",
    "metric_model_1d": "partial_coherence_1d",
    "metric_model_2d": "partial_coherence_2d_qdht",
}


def _choose_n_modes(sigma_s: float, sigma_g: float) -> int:
    a = 1.0 / (4.0 * sigma_s ** 2)
    b = 1.0 / (2.0 * sigma_g ** 2)
    c = math.sqrt(a * (a + 2.0 * b))
    beta_ratio = b / (a + b + c)
    if beta_ratio <= 0.0:
        return 1
    if beta_ratio >= 1.0:
        return MAX_MODES
    n_modes = int(math.ceil(math.log(TAIL_TOLERANCE) / math.log(beta_ratio)))
    return max(1, min(MAX_MODES, n_modes))


def _build_n_modes_grid() -> np.ndarray:
    sigma_g_vals = np.asarray(SWEEP_AXES["sigma_g"])
    sigma_s_vals = np.asarray(SWEEP_AXES["sigma_s"])
    grid = np.empty((sigma_g_vals.size, sigma_s_vals.size), dtype=np.int64)
    for i, sg in enumerate(sigma_g_vals):
        for j, ss in enumerate(sigma_s_vals):
            grid[i, j] = _choose_n_modes(sigma_s=float(ss), sigma_g=float(sg))
    return grid


N_MODES_GRID = _build_n_modes_grid()
PARAM_OVERRIDES["n_modes_grid"] = N_MODES_GRID
PARAM_OVERRIDES["n_modes_grid_axes"] = ("sigma_g", "sigma_s")


def build_point_overrides(index_tuple, axis_values, base_params):
    sigma_s = float(axis_values["sigma_s"])
    sigma_g = float(axis_values["sigma_g"])
    return {
        "sigma_s": sigma_s,
        "sigma_g": sigma_g,
        "n_modes": int(N_MODES_GRID[index_tuple]),
        "optimization_model": "partial_coherence_1d",
        "metric_model_1d": "partial_coherence_1d",
        "metric_model_2d": "partial_coherence_2d_qdht",
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(params["n_modes"])
