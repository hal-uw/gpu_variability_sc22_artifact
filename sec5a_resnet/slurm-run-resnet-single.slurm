#!/bin/bash
#SBATCH -J resnet              # Job name
#SBATCH -o debug/myresnet.%j.out     # Name of stdout output file
#SBATCH -e debug/myresnet.%j.err     # Name of stderr error file
#SBATCH -p gpu-a100                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=kchen346@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
node=$SLURM_JOB_NODELIST
module load tacc-apptainer
echo "Node Number: ${node}"
./resnet-singularity-single.sh
