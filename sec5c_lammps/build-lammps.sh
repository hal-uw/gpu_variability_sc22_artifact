#!/bin/bash

cd src
make clean-all
make yes-user-reaxc
make yes-kokkos
make -j kokkos_cuda_mpi KOKKOS_ARCH=Volta70
