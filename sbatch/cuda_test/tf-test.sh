#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=tf-test.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module load mamba
source activate openvla-spot

nvidia-smi

lspci | grep -i nvidia

# For the next line to work, you need to be in the
# hpc-examples directory.
srun python tf-test.py
