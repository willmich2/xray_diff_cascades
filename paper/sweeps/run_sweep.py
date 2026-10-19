import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.reproduce import iter_studies, resolve_study
from paper.sweeps.sweep_framework import SweepRuntimeConfig, run_sweep
from src import console


def _print_available_studies() -> None:
    print("Available paper sweep studies:\n")
    for study in iter_studies():
        alias_text = ", ".join(study.aliases) if study.aliases else "-"
        notebooks_text = ", ".join(study.notebooks)
        print(f"- {study.key}")
        print(f"    description: {study.description}")
        print(f"    manuscript:  {study.manuscript_result}")
        print(f"    notebooks:   {notebooks_text}")
        print(f"    aliases:     {alias_text}")
        print(f"    config:      {study.config_module}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic optimization sweep runner")
    selector_group = parser.add_mutually_exclusive_group(required=False)
    selector_group.add_argument("--config", help="Python module path for sweep config")
    selector_group.add_argument("--study", help="Friendly paper sweep key (or alias)")
    parser.add_argument("--list-studies", action="store_true", help="Print available paper studies and exit")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--timestamp-outputs", action="store_true", help="Write timestamped files instead of the bundled names.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_studies:
        _print_available_studies()
        return

    config_module = args.config
    if args.study:
        try:
            study = resolve_study(args.study)
        except KeyError as exc:
            parser.error(str(exc))
        config_module = study.config_module
        console.info("run_sweep", f"resolved study '{args.study}' -> {config_module}")

    if not config_module:
        parser.error("One of --config or --study is required unless --list-studies is set.")

    start = console.script_start("run_sweep")
    console.kv("run_sweep", "config_module", config_module)
    console.kv("run_sweep", "save_dir", args.save_dir or "(from config/env)")
    console.kv("run_sweep", "dry_run", args.dry_run)

    runtime = SweepRuntimeConfig(
        config_module=config_module,
        max_workers=args.max_workers,
        workers_per_gpu=args.workers_per_gpu,
        n_runs=args.n_runs,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
        timestamp_outputs=args.timestamp_outputs,
    )
    run_sweep(runtime)
    console.script_done("run_sweep", start)


if __name__ == "__main__":
    main()
