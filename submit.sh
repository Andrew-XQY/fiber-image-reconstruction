#!/bin/bash
# submit.sh — PyTorch 2.5 + CUDA 12.1 (Apptainer)
#
# Usage (run once to prepare, then submit):
#   cd ~
#   # (If needed) pull the container image:
#   # apptainer pull torch_2.5_py312.sif docker://ghcr.io/andrew-xqy/ml-containers:2.5-py3.12-cuda12.1
#   mkdir -p ~/dataset ~/results ~/results/slurm ~/code
#
# Check which --partition are available on my cluster:
#   sinfo
#
# Submit and running the job:
#   sbatch submit.sh
#
# Quick GPU test (optional):
#   apptainer exec --nv ~/torch_2.5_py312.sif python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
#
# Check sbatch detailed info:
#   scontrol show job <id>
#
# Download results e.g.
# scp -r qiyuanxu@barklaviz2.liv.ac.uk:/users/qiyuanxu/results /Users/andrewxu/Desktop/HPC

#SBATCH --job-name=beam_image_reconstruction_model_training
#SBATCH --output=results/slurm/training-%j.out
#SBATCH --error=results/slurm/training-%j.err
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
module load apptainer

IMG="$HOME/torch_2.5_py312.sif"
DATA="$HOME/dataset"      # must contain a subfolder MMF
OUT="$HOME/results"
CODE="$HOME/code"

echo "Using container image: $IMG"  # for debug check

# Match SBATCH log path (relative to submit dir)
mkdir -p results/slurm

srun apptainer exec --nv --writable-tmpfs \
  -B "$DATA:/workspace/dataset:ro" \
  -B "$OUT:/workspace/results" \
  -B "$CODE:/workspace/code" \
  --env MACHINE=liverpool-hpc \
  "$IMG" /bin/bash -lc '
    set -e
    python -m pip install -U pip
    if [ -d /workspace/code/XFlow ]; then pip install -e /workspace/code/XFlow; fi
    cd /workspace/code/examples/fiber-image-reconstruction-comparison
    python train.py
  '