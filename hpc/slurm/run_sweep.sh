#!/bin/bash

#SBATCH --gres=gpu:volta:2
#SBATCH -n 40
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch run_sweep.sh <study_name> [extra run_sweep.py args...]"
  echo
  echo "Examples:"
  echo "  sbatch run_sweep.sh n_sweeps"
  echo "  sbatch run_sweep.sh bandwidth_energy --workers-per-gpu 2"
  echo "  sbatch run_sweep.sh thickness_energy --save-dir paper_data"
  echo
  echo "List studies with:"
  echo "  python paper/sweeps/run_sweep.py --list-studies"
  exit 1
fi

STUDY_NAME="$1"
shift

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh"
  elif [[ -f "${SLURM_SUBMIT_DIR}/_gpu_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/_gpu_env.sh"
  else
    echo "Could not find _gpu_env.sh under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
  fi
else
  source "$(cd "$(dirname "$0")" && pwd)/_gpu_env.sh"
fi

echo "Running paper sweep study: ${STUDY_NAME}"
python paper/sweeps/run_sweep.py --study "${STUDY_NAME}" "$@"
