#!/bin/bash
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
#SBATCH --ntasks=64        # number of MPI tasks, abbreviated by -n

# additional information for allocated clusters
#SBATCH --account=usucs5030   # account, abbreviated by -A
#SBATCH --partition=kingspeak  # partition, abbreviated by -p

#
# set data and working directories
cd /uufs/chpc.utah.edu/common/home/u6039417/CS6030/Homework/5_mpi_histogram

module load openmpi

# run the program
python3 run.py
