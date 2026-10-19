"""Filename and orchestration checks; no optimization or GPU work is performed."""
import argparse
import importlib
import os
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from paper import reproduce
from paper.data_paths import resolve_fig1_n_sweep_paths, robustness_results_path
from paper.sweeps import sweep_framework as sweeps
from paper.reproduce import iter_studies, resolve_study

ROOT = Path(__file__).resolve().parents[1]


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, task):
        future = Future()
        future.set_result(fn(task))
        return future


class PaperPipelineTests(unittest.TestCase):
    def test_renamed_thickness_study_keeps_grid_and_aliases(self):
        study = resolve_study("thickness_energy")
        self.assertEqual(study.config_module, "paper.sweeps.configs.thickness_energy_sweep")
        for alias in ("thickness_energy_fig2a", "thickness_energy_main", "thickness", "aspect_ratio"):
            self.assertIs(resolve_study(alias), study)
        config = importlib.import_module(study.config_module)
        bundled = np.load(ROOT / "paper_data" / f"{config.SAVE_PREFIX}_sweep_arrays.npy", allow_pickle=True).item()
        for axis, values in config.SWEEP_AXES.items():
            self.assertEqual(len(values), 30)
            np.testing.assert_allclose(values, bundled[axis])
        self.assertEqual(reproduce.TARGET_BY_KEY["fig2c"].value, study.key)
        self.assertEqual(reproduce.TARGET_BY_KEY["fig2a_thickness"].value, study.key)

    def test_registered_paths_and_bundled_names(self):
        for target in reproduce.TARGETS:
            for notebook in target.notebook.split(", "):
                self.assertTrue((ROOT / notebook).is_file(), notebook)
            if target.kind == "script":
                self.assertTrue((ROOT / target.value).is_file(), target.value)
            else:
                resolve_study(target.value)
        for study in iter_studies():
            config = importlib.import_module(study.config_module)
            prefix = config.SAVE_PREFIX
            for kind in ("params", "sweep_arrays"):
                self.assertTrue((ROOT / "paper_data" / f"{prefix}_{kind}.npy").is_file())
            suffix = "_run_0" if getattr(config, "SAVE_RUN_SUFFIX", True) or config.N_RUNS > 1 else ""
            self.assertTrue((ROOT / "paper_data" / f"{prefix}_results{suffix}.npz").is_file())

    def test_sweep_writes_stable_and_timestamped_datasets(self):
        cases = ((1, False, False), (1, True, False), (2, False, False), (1, False, True))
        for n_runs, run_suffix, timestamped in cases:
            with self.subTest(n_runs=n_runs, run_suffix=run_suffix, timestamped=timestamped):
                config = {
                    "SAVE_PREFIX": "figS5b_fzp_cascade_thickness_energy_sweep",
                    "N_RUNS": n_runs,
                    "SAVE_RUN_SUFFIX": run_suffix,
                    "worker_fn": lambda task: {"status": "ok", "task_id": task["task_id"], "result": {}},
                }
                with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS": "0", "DIFFRACTIVE_CASCADES_QUIET": "1"}), patch.multiple(
                    sweeps,
                    _build_base_params=Mock(return_value={}),
                    _build_tasks=Mock(return_value=([{"task_id": 1}], config, {"energies": np.array([10000.0])})),
                    _prepare_params_for_save=Mock(return_value={"Nx": 2}),
                    _collect_results=Mock(return_value={"opt_efficiencies": np.array([0.5])}),
                    ProcessPoolExecutor=InlinePool,
                    get_formatted_datetime=lambda: "20261019_120000",
                ), patch.object(sweeps.mp, "get_context"), patch.object(sweeps.torch.cuda, "is_available", return_value=False):
                    sweeps.run_sweep(sweeps.SweepRuntimeConfig(
                        config_module="unused", save_dir=directory, max_workers=1,
                        timestamp_outputs=timestamped,
                    ))
                    prefix = config["SAVE_PREFIX"]
                    stamp = "_20261019_120000" if timestamped else ""
                    expected = {f"{prefix}_params{stamp}.npy", f"{prefix}_sweep_arrays{stamp}.npy"}
                    for run in range(n_runs):
                        suffix = f"_run_{run}" if run_suffix or n_runs > 1 else ""
                        name = f"{prefix}_results{stamp}{suffix}.npz"
                        expected.add(name)
                        with np.load(Path(directory) / name) as result:
                            np.testing.assert_allclose(result["opt_efficiencies"], [0.5])
                    self.assertEqual({p.name for p in Path(directory).iterdir()}, expected)

    def test_robustness_paths_preserve_legacy_input_selection(self):
        paths = resolve_fig1_n_sweep_paths("data", 2, "old_run")
        self.assertEqual(paths[0].name, "fig1_N_sweeps_results_old_run_run_2.npz")
        self.assertEqual(paths[1].name, "fig1_N_sweeps_params_old_run.npy")
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS": enabled}):
                path = robustness_results_path("data", "figS4c_tilt_robustness_results", "old_run", "new_run")
                suffix = "_old_run_new_run" if enabled == "1" else ""
                self.assertEqual(path.name, f"figS4c_tilt_robustness_results{suffix}.npz")

    def test_fzp_checkpoints_write_notebook_compatible_names(self):
        from paper.experiments.fzp_cascade_nelem_sweep import _save_outputs

        for enabled in ("0", "1"):
            with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS": enabled}
            ):
                params, results = _save_outputs(
                    directory, "20261019_120000", {"Nelem": 5},
                    {"opt_efficiencies": np.array([0.5])},
                )
                suffix = "_20261019_120000" if enabled == "1" else ""
                prefix = "figS5a_fzp_cascade_nelem_sweep"
                self.assertEqual(Path(params).name, f"{prefix}_params{suffix}.npy")
                self.assertEqual(Path(results).name, f"{prefix}_results{suffix}.npz")
                self.assertEqual(np.load(params, allow_pickle=True).item()["Nelem"], 5)
                with np.load(results) as saved:
                    np.testing.assert_allclose(saved["opt_efficiencies"], [0.5])

    def test_shared_targets_run_once_and_aliases_still_work(self):
        args = argparse.Namespace(
            action="run", targets=["fig1c", "fig1d", "fig1e", "figS2", "fig2c", "fig2a_thickness", "figA1", "figS1"],
            data_dir="temporary-data", workers_per_gpu=None, max_workers=None,
            n_runs=None, base_n_sweep_id=None, n_sweep_run_id=0,
            timestamp_outputs=True, dry_run=False,
        )
        with patch.object(reproduce, "_parse_args", return_value=args), patch.object(reproduce.subprocess, "run") as run, patch.dict(os.environ, {"DIFFRACTIVE_CASCADES_QUIET": "1"}):
            reproduce.main()
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0][3] for call in run.call_args_list], ["n_sweeps", "thickness_energy", "coherence_illumination"])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS"], "1")
            self.assertEqual(call.kwargs["env"]["DIFFRACTIVE_CASCADES_DATA_DIR"], "temporary-data")


if __name__ == "__main__":
    unittest.main()
