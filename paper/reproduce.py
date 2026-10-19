from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import console
from paper.data_paths import DEFAULT_DATA_DIR


DEFAULT_ROBUSTNESS_BASE_ID: str | None = None
DEFAULT_ROBUSTNESS_RUN_ID = 0


@dataclass(frozen=True)
class Target:
    key: str
    manuscript_result: str
    description: str
    notebook: str
    kind: str  # "sweep" or "script"
    value: str  # sweep study key or script path


@dataclass(frozen=True)
class Figure:
    key: str
    manuscript_result: str
    notebook: str


@dataclass(frozen=True)
class SweepStudy:
    key: str
    config_module: str
    description: str
    figures: tuple[Figure, ...]
    aliases: tuple[str, ...] = ()

    @property
    def manuscript_result(self) -> str:
        return ", ".join(dict.fromkeys(figure.manuscript_result for figure in self.figures))

    @property
    def notebooks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(figure.notebook for figure in self.figures))


# One source of sweep metadata for the reproduction and direct-sweep CLIs.
PAPER_SWEEP_STUDIES: tuple[SweepStudy, ...] = (
    SweepStudy(
        key="n_sweeps",
        config_module="paper.sweeps.configs.n_sweeps",
        description="Material and cascade-depth scaling",
        figures=(
            Figure("fig1e", "Fig. 1(e)", "notebooks/fig1e_Nelem_sweep.ipynb"),
            Figure("fig1c", "Fig. 1(c)", "notebooks/fig1c_element_visualization.ipynb"),
            Figure("fig1d", "Fig. 1(d)", "notebooks/fig1d_focal_spot_comparison.ipynb"),
            Figure("figS2", "Fig. S2", "notebooks/figS2_physical_bound.ipynb"),
        ),
        aliases=("n", "nelem"),
    ),
    SweepStudy(
        key="bandwidth_energy",
        config_module="paper.sweeps.configs.bandwidth_energy_sweep",
        description="Bandwidth vs energy map",
        figures=(
            Figure("fig2a_bandwidth", "Fig. 2(a)", "notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb"),
        ),
        aliases=("bandwidth", "bw"),
    ),
    SweepStudy(
        key="thickness_energy",
        config_module="paper.sweeps.configs.thickness_energy_sweep",
        description="Shared 30x30 thickness-energy grid for Figs. 2(a) and 2(c)",
        figures=(
            Figure("fig2a_thickness", "Fig. 2(a)", "notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb"),
            Figure("fig2c", "Fig. 2(c)", "notebooks/fig2c_aspect_ratio_scaling.ipynb"),
        ),
        aliases=(
            "thickness_energy_dense", "thickness_fig2a", "thickness", "aspect_ratio",
            "thickness_energy_fig2a", "thickness_energy_main",
        ),
    ),
    SweepStudy(
        key="nelem_min_feature",
        config_module="paper.sweeps.configs.nelem_min_feature_sweep",
        description="Spot-size vs efficiency tradeoff",
        figures=(
            Figure("fig2b", "Fig. 2(b)", "notebooks/fig2b_mfs_sweep.ipynb"),
        ),
        aliases=("mfs", "min_feature"),
    ),
    SweepStudy(
        key="fzp_cascade_bandwidth_energy",
        config_module="paper.sweeps.configs.fzp_cascade_bandwidth_energy_sweep",
        description="Bandwidth vs energy map for a 10-element FZP cascade vs optimized cascade",
        figures=(
            Figure("figS5b_bandwidth", "Fig. S5(b), bandwidth-energy", "notebooks/figS5b_fzp_cascade_energy_bandwidth_aspect_ratio.ipynb"),
        ),
        aliases=("fzp_cascade_bandwidth", "fzp_cascade_bw"),
    ),
    SweepStudy(
        key="fzp_cascade_thickness_energy",
        config_module="paper.sweeps.configs.fzp_cascade_thickness_energy_sweep",
        description="Thickness vs energy map for a 10-element FZP cascade vs optimized cascade",
        figures=(
            Figure("figS5b_thickness", "Fig. S5(b), thickness-energy", "notebooks/figS5b_fzp_cascade_energy_bandwidth_aspect_ratio.ipynb"),
        ),
        aliases=("fzp_cascade_thickness", "fzp_cascade_aspect_ratio"),
    ),
    SweepStudy(
        key="coherence_illumination",
        config_module="paper.sweeps.configs.coherence_illumination_sweep",
        description="Partial spatial coherence sweep",
        figures=(
            Figure("figS1", "Fig. S1", "notebooks/figS1_partial_coherence.ipynb"),
        ),
        aliases=("coherence",),
    ),
    SweepStudy(
        key="focal_length",
        config_module="paper.sweeps.configs.focal_length_sweeps",
        description="Focal-length scaling",
        figures=(
            Figure("figS3_focal", "Fig. S3(a)", "notebooks/figS3a_focal_length.ipynb"),
        ),
        aliases=("focal",),
    ),
    SweepStudy(
        key="inter_element_distance",
        config_module="paper.sweeps.configs.inter_elem_dist_sweeps",
        description="Inter-element spacing scaling",
        figures=(
            Figure("figS3_inter", "Fig. S3(b)", "notebooks/figS3b_inter_elem_dist.ipynb"),
        ),
        aliases=("inter_elem_dist", "inter_elem"),
    ),
)


_STUDY_BY_ALIAS = {
    alias: study for study in PAPER_SWEEP_STUDIES for alias in (study.key, *study.aliases)
}


def iter_studies() -> tuple[SweepStudy, ...]:
    return PAPER_SWEEP_STUDIES


def resolve_study(study_name: str) -> SweepStudy:
    key = study_name.strip()
    if key not in _STUDY_BY_ALIAS:
        valid = ", ".join(sorted(_STUDY_BY_ALIAS))
        raise KeyError(f"Unknown sweep study '{study_name}'. Valid studies/aliases: {valid}")
    return _STUDY_BY_ALIAS[key]


SCRIPT_TARGETS: tuple[Target, ...] = (
    Target(
        key="fig3_placement",
        manuscript_result="Fig. 3(d)",
        description="Placement robustness postprocess",
        notebook="notebooks/fig3d_robustness.ipynb",
        kind="script",
        value="paper/postprocess/placement_robustness.py",
    ),
    Target(
        key="fig3_erosion_dilation",
        manuscript_result="Fig. 3(d), Fig. S4(a)",
        description="Erosion/dilation robustness postprocess",
        notebook="notebooks/fig3d_robustness.ipynb, notebooks/figS4_robustness.ipynb",
        kind="script",
        value="paper/postprocess/erosion_dilation_robustness.py",
    ),
    Target(
        key="fig3_thermal",
        manuscript_result="Fig. 3(d)",
        description="Thermal robustness postprocess",
        notebook="notebooks/fig3d_robustness.ipynb",
        kind="script",
        value="paper/postprocess/thermal_robustness.py",
    ),
    Target(
        key="figS4_sidewall",
        manuscript_result="Fig. S4(b)",
        description="Sidewall smoothness robustness postprocess",
        notebook="notebooks/figS4_robustness.ipynb",
        kind="script",
        value="paper/postprocess/sidewall_robustness.py",
    ),
    Target(
        key="figS4_tilt",
        manuscript_result="Fig. S4(c)",
        description="Element-tilt robustness postprocess",
        notebook="notebooks/figS4_robustness.ipynb",
        kind="script",
        value="paper/postprocess/tilt_robustness.py",
    ),
    Target(
        key="figS5a",
        manuscript_result="Fig. S5(a)",
        description="Optimized cascade vs FZP cascade across element counts",
        notebook="notebooks/figS5a_fzp_cascade_nelem_sweep.ipynb",
        kind="script",
        value="paper/experiments/fzp_cascade_nelem_sweep.py",
    ),
    Target(
        key="fig4b",
        manuscript_result="Fig. 4(b)",
        description="Single-plane vs multi-plane depth-of-focus experiment",
        notebook="notebooks/fig4b_depth_of_focus.ipynb",
        kind="script",
        value="paper/experiments/xray_focusing_focal_sweep_comparison.py",
    ),
)

TARGETS: tuple[Target, ...] = tuple(
    Target(figure.key, figure.manuscript_result, study.description, figure.notebook, "sweep", study.key)
    for study in PAPER_SWEEP_STUDIES
    for figure in study.figures
) + SCRIPT_TARGETS

TARGET_BY_KEY = {target.key: target for target in TARGETS}
# Keep existing commands working while the listed keys follow the current figures.
TARGET_ALIASES = {
    "figA1": "figS1",
    "figA3_focal": "figS3_focal",
    "figA3_inter": "figS3_inter",
    "figA4_sidewall": "figS4_sidewall",
    "fzp_cascade_bandwidth": "figS5b_bandwidth",
    "fzp_cascade_thickness": "figS5b_thickness",
}
TARGET_BY_KEY.update({alias: TARGET_BY_KEY[key] for alias, key in TARGET_ALIASES.items()})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper reproduction helper for diffractive cascades.")
    parser.add_argument("action", choices=("list", "run"))
    parser.add_argument(
        "targets",
        nargs="*",
        help="Target keys to run. Use 'all' to run every target from `list`.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument(
        "--base-n-sweep-id",
        default=DEFAULT_ROBUSTNESS_BASE_ID,
        help="fig1_N_sweeps timestamp for postprocess inputs; omit for bundled paper_data/ files.",
    )
    parser.add_argument("--n-sweep-run-id", type=int, default=DEFAULT_ROBUSTNESS_RUN_ID)
    parser.add_argument("--timestamp-outputs", action="store_true", help="Keep separately timestamped outputs; notebooks use stable names by default.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _print_targets() -> None:
    print("Paper reproduction targets:\n")
    for target in TARGETS:
        print(f"- {target.key}: {target.manuscript_result}")
        print(f"    description: {target.description}")
        print(f"    notebook:    {target.notebook}")
        if target.kind == "sweep":
            print(f"    command:     python paper/sweeps/run_sweep.py --study {target.value}")
        else:
            print(f"    command:     python {target.value}")
        print()


def _build_command(target: Target, args: argparse.Namespace) -> list[str]:
    if target.kind == "sweep":
        cmd = [sys.executable, "paper/sweeps/run_sweep.py", "--study", target.value]
        if args.data_dir:
            cmd += ["--save-dir", args.data_dir]
        if args.workers_per_gpu is not None:
            cmd += ["--workers-per-gpu", str(args.workers_per_gpu)]
        if args.max_workers is not None:
            cmd += ["--max-workers", str(args.max_workers)]
        if args.n_runs is not None:
            cmd += ["--n-runs", str(args.n_runs)]
        return cmd

    cmd = [sys.executable, target.value]
    if "postprocess" in target.value:
        if args.base_n_sweep_id:
            cmd += ["--base-id", args.base_n_sweep_id]
        cmd += ["--run-id", str(args.n_sweep_run_id)]
        if args.workers_per_gpu is not None:
            cmd += ["--workers-per-gpu", str(args.workers_per_gpu)]
        if args.data_dir:
            cmd += ["--data-dir", args.data_dir]
    elif target.key == "figS5a":
        if args.n_runs is not None:
            cmd += ["--n-runs", str(args.n_runs)]
        if args.data_dir:
            cmd += ["--save-dir", args.data_dir]
    return cmd


def main() -> None:
    args = _parse_args()
    if args.action == "list":
        _print_targets()
        return

    if not args.targets:
        raise SystemExit("No targets specified. Use `list` to see available keys.")

    start = console.script_start("reproduce", argv=sys.argv[1:])
    console.kv("reproduce", "data_dir", args.data_dir)
    console.kv("reproduce", "dry_run", args.dry_run)

    selected_keys = [target.key for target in TARGETS] if "all" in args.targets else args.targets
    unknown = [key for key in selected_keys if key not in TARGET_BY_KEY]
    if unknown:
        valid = ", ".join(sorted(TARGET_BY_KEY.keys()))
        raise SystemExit(f"Unknown target(s): {', '.join(unknown)}. Valid keys: {valid}")

    env = os.environ.copy()
    if args.timestamp_outputs:
        env["DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS"] = "1"
    if args.data_dir:
        env["DIFFRACTIVE_CASCADES_DATA_DIR"] = args.data_dir

    console.info("reproduce", f"running {len(selected_keys)} target(s): {', '.join(selected_keys)}")
    completed_jobs = set()
    for key in selected_keys:
        target = TARGET_BY_KEY[key]
        job = (target.kind, target.value)
        if job in completed_jobs:
            console.info("reproduce", f"{target.key}: reuses an already selected job")
            continue
        completed_jobs.add(job)
        cmd = _build_command(target, args)
        console.banner("reproduce", f"{target.key}: {target.manuscript_result}")
        console.info("reproduce", target.description)
        console.command("reproduce", cmd)
        if args.dry_run:
            console.info("reproduce", "dry-run: skipping subprocess execution")
            continue
        target_start = time.time()
        subprocess.run(cmd, check=True, env=env)
        console.elapsed("reproduce", f"target {target.key} complete", time.time() - target_start)

    console.script_done("reproduce", start)


if __name__ == "__main__":
    main()
