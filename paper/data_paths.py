"""Shared naming for paper datasets and robustness inputs.

Outputs use the bundled, notebook-compatible names by default. Set
DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS=1 to keep separately timestamped runs.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "paper_data")
DEFAULT_BASE_SWEEP_ID: str | None = None
FIG1_N_SWEEPS_PREFIX = "fig1_N_sweeps"


def dataset_path(
    data_dir: str | os.PathLike[str],
    prefix: str,
    kind: str,
    *,
    timestamp: str | None = None,
    run_id: int | None = None,
) -> Path:
    """Build a params, sweep_arrays, or results path, including optional run IDs."""
    if kind not in ("params", "sweep_arrays", "results"):
        raise ValueError(f"Unknown dataset kind: {kind}")
    suffix = f"_{timestamp}" if timestamp else ""
    if run_id is not None:
        suffix += f"_run_{run_id}"
    extension = "npz" if kind == "results" else "npy"
    return Path(data_dir) / f"{prefix}_{kind}{suffix}.{extension}"


def output_timestamp(save_time: str, *, enabled: bool = False) -> str | None:
    """Choose stable output names unless timestamped output was requested."""
    if enabled or os.environ.get("DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS") == "1":
        return save_time
    return None


def resolve_fig1_n_sweep_paths(
    data_dir: str | os.PathLike[str],
    run_id: int,
    base_id: str | None = None,
) -> tuple[Path, Path, Path]:
    """Return (results, params, sweep_arrays) for bundled or timestamped Fig. 1 inputs."""
    return (
        dataset_path(data_dir, FIG1_N_SWEEPS_PREFIX, "results", timestamp=base_id, run_id=run_id),
        dataset_path(data_dir, FIG1_N_SWEEPS_PREFIX, "params", timestamp=base_id),
        dataset_path(data_dir, FIG1_N_SWEEPS_PREFIX, "sweep_arrays", timestamp=base_id),
    )


def output_id_label(base_id: str | None) -> str:
    """Value stored in robustness output NPZ ID metadata."""
    return base_id or FIG1_N_SWEEPS_PREFIX


def robustness_results_path(
    data_dir: str | os.PathLike[str],
    output_prefix: str,
    base_id: str | None,
    save_time: str,
) -> Path:
    """Use notebook names by default, retaining input IDs in timestamped output names."""
    timestamp = output_timestamp(save_time)
    if timestamp and base_id:
        timestamp = f"{base_id}_{timestamp}"
    suffix = f"_{timestamp}" if timestamp else ""
    return Path(data_dir) / f"{output_prefix}{suffix}.npz"
