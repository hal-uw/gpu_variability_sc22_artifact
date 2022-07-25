#!/bin/bash

cd src
export CUDA_HOME=/usr/local/cuda-10.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
make clean-all
make yes-user-reaxc
make yes-kokkos
make -j kokkos_cuda_mpi KOKKOS_ARCH=Volta70
