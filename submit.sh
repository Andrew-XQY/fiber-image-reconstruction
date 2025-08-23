#!/bin/bash
# submit.sh — PyTorch 2.5 + CUDA 12.1 (Apptainer)
#
# Usage (run once to prepare, then submit):
#   cd ~
#   # (If needed) pull the container image:
#   # apptainer pull torch_2.5_py312.sif docker://ghcr.io/andrew-xqy/ml-containers:2.5-py3.12-cuda12.1
#   mkdir -p ~/dataset ~/results ~/results/slurm ~/code
#
#   sbatch submit.sh
#
# Quick GPU test (optional):
#   apptainer exec --nv ~/torch_2.5_py312.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

#SBATCH --job-name=beam_image_reconstruction_model_training
#SBATCH --output=results/slurm/training-%j.out
#SBATCH --error=results/slurm/training-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
module load apps/apptainer

IMG="$HOME/torch_2.5_py312.sif"
DATA="$HOME/dataset"      # must contain a subfolder MMF
OUT="$HOME/results"
CODE="$HOME/code"

# Match SBATCH log path (relative to submit dir)
mkdir -p results/slurm

srun apptainer exec --nv --writable-tmpfs \
  -B "$DATA:/workspace/dataset:ro" \
  # If you prefer stricter binding, use this instead:
  # -B "$DATA/MMF:/workspace/dataset/MMF:ro" \
  -B "$OUT:/workspace/results" \
  -B "$CODE:/workspace/code" \
  --env MACHINE=liverpool-hpc \
  --env EXPERIMENT_NAME=CAE_run1 \
  --env PROJECT_ROOT=/workspace/code \
  "$IMG" /bin/bash -lc '
    set -e
    python -m pip install -U pip
    if [ -d /workspace/code/xflow ]; then pip install -e /workspace/code/xflow; fi
    cd /workspace/code
    python examples/train.py
  '