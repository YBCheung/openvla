#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --partition=gpu-h100-80g
#SBATCH --output=sbatch/1finetune.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module load mamba
source activate openvla-spot

nvidia-smi

pwd

# # check GPU related versions
# nvcc --version  # System CUDA compiler version
# echo "Pytorch ≥1.12, CUDA toolkit ≥11.0 for stable bfloat16 support"
# python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())" 
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py 
