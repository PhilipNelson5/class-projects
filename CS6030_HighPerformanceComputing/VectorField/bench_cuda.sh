#!/bin/bash
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
#SBATCH --ntasks=1        # number of MPI tasks, abbreviated by -n

# additional information for allocated clusters
#SBATCH --account=notchpeak-shared-short   # account, abbreviated by -A
#SBATCH --partition=notchpeak-shared-short  # partition, abbreviated by -p
#SBATCH --gres=gpu:k80:1

# set data and working directories
cd /uufs/chpc.utah.edu/common/home/u6039417/CS6030/VectorField

# load required modules
module load cuda/10

# run the program
python3 bench_cuda.py
