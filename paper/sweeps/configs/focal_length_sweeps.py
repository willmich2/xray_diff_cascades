import numpy as np

from paper.sweeps.standard_params import NX_DEFAULT

SAVE_PREFIX = "figS3a_focal_length_sweeps"
SAVE_DIR = None
N_RUNS = 1
SAVE_RUN_SUFFIX = True
PARAM_OVERRIDES = {
    "efficiency_normalization": "active_device_area",
}
SWEEP_AXES = {
    "Nelems": np.array([5, 10, 20, 40], dtype=int),
    "focal_lengths": np.logspace(-3, -0.5, 8),
}
MAX_PARAMS = int(int(np.max(SWEEP_AXES["Nelems"])) * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)


def build_point_overrides(index_tuple, axis_values, base_params):
    return {
        "Nelem": int(axis_values["Nelems"]),
        "f": float(axis_values["focal_lengths"]),
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(axis_values["Nelems"])
