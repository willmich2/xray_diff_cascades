import numpy as np

from paper.sweeps.standard_params import NX_DEFAULT

SAVE_PREFIX = "fig2b_Nelem_min_feature_sweep"
SAVE_DIR = None
N_RUNS = 3
PARAM_OVERRIDES = {
    "efficiency_normalization": "active_device_area",
}
SWEEP_AXES = {
    "Nelem": np.array([5, 10, 20, 40], dtype=int),
    "min_feature_size": np.linspace(10e-9, 200e-9, 10, dtype=float),
}
MAX_PARAMS = int(int(np.max(SWEEP_AXES["Nelem"])) * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)


def build_point_overrides(index_tuple, axis_values, base_params):
    min_feature = float(axis_values["min_feature_size"])
    return {
        "Nelem": int(axis_values["Nelem"]),
        "min_feature_size": min_feature,
        "element_thickness": 8.0 * min_feature,
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(axis_values["Nelem"])
