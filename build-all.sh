#!/bin/bash

# Builds necessary for running all 5 experiments:
# SGEMM on Nvidia GPUs
# SGEMM on AMD GPUs
# ResNet-50

# LAMMPS
echo "Building packages"
cd sec5b_lammps/lammps_17Jan18/src
./build-lammps.sh
echo "Compiling binary assuming ARCH=Volta70 - to change, edit sec5b_lammps/lammps_17Jan18/src/build-lammps.sh"
cd ../../../
echo "Done building packages for LAMMPS"

echo ${PWD}

# PageRank
echo "Compiling PageRank SPMV" 
cd sec5c_pagerank/gpu-app-collection
./build-pagerank.sh
echo "Done compiling binary for Pagerank"
echo "Build Completed"
