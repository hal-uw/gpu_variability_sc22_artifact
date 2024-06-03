#!/bin/bash
#SBATCH -J pagerank              # Job name
#SBATCH -o debug/mypagerank.%j.out     # Name of stdout output file
#SBATCH -e debug/mypagerank.%j.err     # Name of stderr error file
#SBATCH -p rtx                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-user=kchen346@wisc.edu
#SBATCH --mail-type=all          # Send email at begin and end of job
node=$SLURM_JOB_NODELIST
module load tacc-apptainer
module load cuda
echo "Node Number: ${node}"
./fetch-input.sh
./build-pagerank.sh
for i in {0..2}; do
    for j in {0..4}; do
        ./run-pagerank.sh $i $j
    done
done
