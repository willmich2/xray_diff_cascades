from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from src import console


DEFAULT_DATA_DIR = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "paper_data")
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


TARGETS: tuple[Target, ...] = (
    Target(
        key="fig1e",
        manuscript_result="Fig. 1(e)",
        description="Material/depth scaling for cascade efficiency",
        notebook="notebooks/fig1e_Nelem_sweep.ipynb",
        kind="sweep",
        value="n_sweeps",
    ),
    Target(
        key="fig1c",
        manuscript_result="Fig. 1(c)",
        description="Element visualizations from the N sweep output",
        notebook="notebooks/fig1c_element_visualization.ipynb",
        kind="sweep",
        value="n_sweeps",
    ),
    Target(
        key="fig2a_bandwidth",
        manuscript_result="Fig. 2(a)",
        description="Bandwidth-energy sweep",
        notebook="notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb",
        kind="sweep",
        value="bandwidth_energy",
    ),
    Target(
        key="fig2a_thickness",
        manuscript_result="Fig. 2(a)",
        description="Thickness-energy sweep (30x30 grid used in Fig. 2(a))",
        notebook="notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb",
        kind="sweep",
        value="thickness_energy_fig2a",
    ),
    Target(
        key="fig2b",
        manuscript_result="Fig. 2(b)",
        description="Minimum feature size vs efficiency/spot tradeoff",
        notebook="notebooks/fig2b_mfs_sweep.ipynb",
        kind="sweep",
        value="nelem_min_feature",
    ),
    Target(
        key="fig2c",
        manuscript_result="Fig. 2(c)",
        description="Aspect-ratio scaling sweep",
        notebook="notebooks/fig2c_aspect_ratio_scaling.ipynb",
        kind="sweep",
        value="thickness_energy_main",
    ),
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
        manuscript_result="Fig. 3(d), Fig. A.4(a)",
        description="Erosion/dilation robustness postprocess",
        notebook="notebooks/fig3d_robustness.ipynb",
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
        key="figA4_sidewall",
        manuscript_result="Fig. A.4(b)",
        description="Sidewall smoothness robustness postprocess",
        notebook="notebooks/fig3d_robustness.ipynb",
        kind="script",
        value="paper/postprocess/sidewall_robustness.py",
    ),
    Target(
        key="fig4b",
        manuscript_result="Fig. 4(b)",
        description="Single-plane vs multi-plane depth-of-focus experiment",
        notebook="notebooks/fig4b_depth_of_focus.ipynb",
        kind="script",
        value="paper/experiments/xray_focusing_focal_sweep_comparison.py",
    ),
    Target(
        key="fzp_cascade",
        manuscript_result="FZP cascade comparison",
        description="Optimized cascade vs intermediate-field FZP cascade vs single FZP",
        notebook="notebooks/fzp_cascade_comparison.ipynb",
        kind="script",
        value="paper/experiments/fzp_cascade_comparison.py",
    ),
    Target(
        key="fzp_cascade_bandwidth",
        manuscript_result="FZP cascade vs optimized cascade (bandwidth-energy)",
        description="Bandwidth-energy sweep: 10-element FZP cascade vs optimized cascade",
        notebook="notebooks/fzp_cascade_energy_bandwidth_aspect_ratio.ipynb",
        kind="sweep",
        value="fzp_cascade_bandwidth_energy",
    ),
    Target(
        key="fzp_cascade_thickness",
        manuscript_result="FZP cascade vs optimized cascade (thickness-energy)",
        description="Thickness-energy sweep: 10-element FZP cascade vs optimized cascade",
        notebook="notebooks/fzp_cascade_energy_bandwidth_aspect_ratio.ipynb",
        kind="sweep",
        value="fzp_cascade_thickness_energy",
    ),
    Target(
        key="figA1",
        manuscript_result="Fig. A.1",
        description="Partial coherence sweep",
        notebook="notebooks/figA1_partial_coherence.ipynb",
        kind="sweep",
        value="coherence_illumination",
    ),
    Target(
        key="figA3_focal",
        manuscript_result="Fig. A.3(a)",
        description="Focal-length sweep",
        notebook="notebooks/figA3a_focal_length.ipynb",
        kind="sweep",
        value="focal_length",
    ),
    Target(
        key="figA3_inter",
        manuscript_result="Fig. A.3(b)",
        description="Inter-element-distance sweep",
        notebook="notebooks/figA3b_inter_elem_dist.ipynb",
        kind="sweep",
        value="inter_element_distance",
    ),
)

TARGET_BY_KEY = {target.key: target for target in TARGETS}


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
    elif target.key == "fzp_cascade" and args.data_dir:
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
    if args.data_dir:
        env["DIFFRACTIVE_CASCADES_DATA_DIR"] = args.data_dir

    console.info("reproduce", f"running {len(selected_keys)} target(s): {', '.join(selected_keys)}")
    for key in selected_keys:
        target = TARGET_BY_KEY[key]
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
