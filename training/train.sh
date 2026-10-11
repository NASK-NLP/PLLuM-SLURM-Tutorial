#!/bin/bash
#SBATCH --job-name=pllum-train
#SBATCH --output=out/training_slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=tutorial
#SBATCH --partition=tutorial
#SBATCH --reservation=tutorial-llm

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
DATASET_DIR="$REPO_DIR/data/tools_dataset.jsonl"
MODEL_ID="CYFRAGOVPL/PLLuM-4B-instruct-2512"
RUN_DIR="$SCRATCH/pllum-slurm-lab"
ADAPTER_DIR="$RUN_DIR/adapter"

source "$RUN_DIR/.venv/bin/activate"
python -m training.train \
  --model "$MODEL_ID" \
  --data-file "$DATASET_DIR" \
  --run-dir "$RUN_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  "$@"
