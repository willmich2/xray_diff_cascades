import numpy as np

from paper.sweeps.fzp_cascade_worker import (
    FZP_INTER_ELEM_DIST_DEFAULT,
    fzp_cascade_vs_opt_worker,
)
from paper.sweeps.standard_params import N_ELEMENTS_DEFAULT, NX_DEFAULT

SAVE_PREFIX = "fzp_cascade_thickness_energy_sweep"
SAVE_DIR = None
N_RUNS = 1
SAVE_RUN_SUFFIX = False

# Same 30x30 grid as Fig. 2(a) thickness-energy so the pcolormesh matches
# notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb.
SWEEP_AXES = {
    "thicknesses": np.logspace(-7.3, -4.8, 30),
    "energies": np.linspace(5e3, 27e3, 30),
}

MAX_PARAMS = int(N_ELEMENTS_DEFAULT * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)

PARAM_OVERRIDES = {
    "Nelem": int(N_ELEMENTS_DEFAULT),
    "fzp_inter_elem_dist": float(FZP_INTER_ELEM_DIST_DEFAULT),
    "fzp_baseline": "cascade",
}

worker_fn = fzp_cascade_vs_opt_worker


def build_point_overrides(index_tuple, axis_values, base_params):
    return {
        "element_thickness": float(axis_values["thicknesses"]),
        "central_energy_ev": float(axis_values["energies"]),
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(axis_values["energies"])
