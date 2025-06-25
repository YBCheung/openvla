#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=sbatch/0generate.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module load mamba
source activate openvla
pwd
date
python scripts/generate.py 
echo "finished at $(date)"
