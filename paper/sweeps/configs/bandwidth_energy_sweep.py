import numpy as np

from paper.sweeps.standard_params import N_ELEMENTS_DEFAULT, NX_DEFAULT

SAVE_PREFIX = "fig2a_bandwidth_energy_sweep"
SAVE_DIR = None
N_RUNS = 1
SAVE_RUN_SUFFIX = False
PARAM_OVERRIDES = {
    "efficiency_normalization": "active_device_area",
}
SWEEP_AXES = {
    "bandwidths": np.logspace(-5, -1.8, 30),
    "energies": np.linspace(5e3, 27e3, 30),
}
MAX_PARAMS = int(N_ELEMENTS_DEFAULT * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)


def build_point_overrides(index_tuple, axis_values, base_params):
    return {
        "bandwidth": float(axis_values["bandwidths"]),
        "central_energy_ev": float(axis_values["energies"]),
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(axis_values["energies"])
