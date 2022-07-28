#!/bin/bash

uarch=$(uname -m)
echo "Pulling container for $uarch"

if [[ "$uarch" == *ppc64le* ]]; then
    echo "Base container image does not support underlying $uarch architecture. Please follow steps in README to Build and Run Without Docker"
else
    #singularity pull docker://pawsey/cuda-mpich-base:3.1.4_cuda10.2-devel_ubuntu18.04
    singularity run --nv docker://pawsey/cuda-mpich-base:3.1.4_cuda10.2-devel_ubuntu18.04 ./build-lammps.sh 
    mpirun -n 1 singularity run --nv docker://pawsey/cuda-mpich-base:3.1.4_cuda10.2-devel_ubuntu18.04 ./run-lammps.sh 0 1 0  
fi

echo "LAMMPS run completed. Output CSV can be found in ./reax_benchmark/ directory."