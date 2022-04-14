#!/bin/bash

# Runs all 5 experiments:
# SGEMM on Nvidia GPUs

# SGEMM on AMD GPUs

# ResNet-50

# LAMMPS
echo "Running LAMMPS..."
source sec5b_lammps/reax_benchmark/src-lammps.sh
cd sec5b_lammps/reax_benchmark

for ((num_run=1; num_run<=2; num_run++))
do
    echo "Run Number: ${num_run}, Device: 0"
    #run-lammps gpu_num run_num node_num
    run-lammps.sh 0 $num_run 0
done
cd ../../
echo "$PWD"
echo "LAMMPS Run Completed" 

# PageRank
echo "Running PageRank SPMV..."
cd sec5c_pagerank/
for ((num_run=1; num_run<=2; num_run++))
do
    echo "Run Number: ${num_run}, Device: 0"
    #run-lammps gpu_num run_num node_num
    run-pagerank.sh 0 $num_run 0
done
cd ../
echo "$PWD"
echo "PageRank Run Completed"
