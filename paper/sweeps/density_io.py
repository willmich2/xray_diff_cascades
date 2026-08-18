"""Binary packing for projected element densities in sweep and experiment outputs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

BINARY_DENSITY_DTYPE = np.bool_
INTENSITY_DTYPE = np.float32

# Explicit element-density keys in sweep/experiment ``.npz`` outputs.
BINARY_DENSITY_KEYS = frozenset({"opt_rhos", "rho_bar", "rho_bar_run2", "fzp_cascade_rhos"})

# Minimal fields needed by fig1 notebooks and robustness postprocess scripts.
FIG1_N_SWEEPS_SAVE_KEYS = (
    "opt_rhos",
    "opt_intensities",
    "fzp_intensities",
    "opt_efficiencies",
    "fzp_efficiencies",
    "fill_factors",
    "run_id",
)


def pack_binary_density(rho: Any) -> np.ndarray:
    """Pack a projected density as bool (solid=True, void=False)."""
    arr = np.asarray(rho)
    if arr.dtype == np.bool_:
        return arr
    return (arr > 0.5).astype(BINARY_DENSITY_DTYPE, copy=False)


def load_opt_rhos(rhos: Any) -> np.ndarray:
    """Normalize ``opt_rhos`` from disk (bool or legacy float) without promoting to float64."""
    return pack_binary_density(rhos)


def density_half_profile(
    rhos: np.ndarray,
    index: tuple | int,
    length: int,
    *,
    dtype: type = np.float64,
) -> np.ndarray:
    """Extract an active half-profile as float for simulation or postprocessing."""
    return np.asarray(load_opt_rhos(rhos)[index][:length], dtype=dtype)


def compute_fill_factors(
    opt_rhos: np.ndarray,
    nelems: np.ndarray,
    *,
    nx: int,
    nelem_axis: int = -1,
) -> np.ndarray:
    """Mean solid fraction over each sweep point's active half-profile."""
    rhos = load_opt_rhos(opt_rhos)
    nelems = np.asarray(nelems, dtype=int)
    nelem_axis_pos = nelem_axis if nelem_axis >= 0 else rhos.ndim - 1 + nelem_axis
    half = int(nx) // 2
    out = np.empty(rhos.shape[:-1], dtype=np.float32)
    idx = [slice(None)] * (rhos.ndim - 1)
    for nelem_idx, n_elem in enumerate(nelems):
        length = int(n_elem) * half
        idx[nelem_axis_pos] = nelem_idx
        out[tuple(idx)] = rhos[tuple(idx + [slice(length)])].mean(axis=-1)
    return out


def _is_binary_density_key(key: str) -> bool:
    return key in BINARY_DENSITY_KEYS or key.endswith("_rhos") or key.startswith("rho_")


def _is_intensity_array(value: Any) -> bool:
    arr = np.asarray(value)
    return arr.ndim >= 1 and arr.dtype.kind in "fc"


def _prepare_sweep_payload(arrays: dict[str, Any], *, keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Normalize sweep arrays for storage (binary densities, float32 intensities)."""
    payload = {k: arrays[k] for k in keys if k in arrays} if keys is not None else dict(arrays)
    for key, value in list(payload.items()):
        if _is_binary_density_key(key):
            payload[key] = load_opt_rhos(value)
        elif "intensit" in key.lower() and _is_intensity_array(value):
            payload[key] = np.asarray(value, dtype=INTENSITY_DTYPE)
        elif key in ("opt_efficiencies", "fzp_efficiencies", "fill_factors"):
            payload[key] = np.asarray(value, dtype=INTENSITY_DTYPE)
    return payload


def save_sweep_results(
    path: str,
    arrays: dict[str, Any],
    *,
    keys: Iterable[str] | None = None,
) -> None:
    """Write sweep outputs; element densities are stored as compressed bool arrays."""
    np.savez_compressed(path, **_prepare_sweep_payload(arrays, keys=keys))


def normalize_sweep_results_file(path: str, *, keys: Iterable[str] | None = None) -> None:
    """Rewrite an on-disk sweep ``.npz`` with binary densities and float32 intensities."""
    data = np.load(path, allow_pickle=True)
    save_sweep_results(path, dict(data), keys=keys)


def slim_sweep_results_file(path: str, *, keys: Iterable[str] | None = FIG1_N_SWEEPS_SAVE_KEYS) -> None:
    """Rewrite an on-disk sweep ``.npz`` with binary densities and fewer fields."""
    normalize_sweep_results_file(path, keys=keys)
