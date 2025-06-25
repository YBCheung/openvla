#!/bin/bash
#SBATCH --output=pytorch-test.out

#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G


echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

module load mamba
source activate openvla
# For the next line to work, you need to be in the
# hpc-examples directory.


python pytorch-test.py

for i in $(seq 2); do
	date
	sleep 1 
done
