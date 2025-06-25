#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=sbatch/0min-test.out
#SBATCH --constraint=ampere

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module load mamba
source activate openvla
pwd
date
python scripts/min_test.py 
echo "finished at $(date)"
