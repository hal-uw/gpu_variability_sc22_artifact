# Characterizing Variability in Large-Scale, Accelerator-Rich Systems

This artifact contains code to reproduce the experiments carried out in "Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems". 

## Table of Contents

- [Experiments](#experiments)
- [Build and Run](#install-build)
- [Related Code](#related)

## Experiments

There were 5 experiments we ran in our paper: 
- SGEMM on NVIDIA V100s: an application we wrote that utilizes NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats 
- SGEMM on AMD MI60s: we also wrote sgemm with AMD's rocBLAS library
- ResNet-50 on NVIDIA V100s: a stable CNN application used widely in the HPC community as a benchmark
- LAMMPS on NVIDIA V100s: a molecular dynamics application widely used for scalable science experiments 
- PageRank on NVIDIA V100s: a graph analytics benchmark that uses spare matrix vector multiplication

For each experiment, there is a corresponding directory in this repository (e.g., SGEMM on NVIDIA: `sec4bc_sgemm_nvidia`). The names of each directory correspond to the sections in our paper.

## Build and Run
To run each of our applications, we provide docker containers in each application directory. These docker containers install all dependencies and compile library code into respective application binaries. Directions to run each application using the docker containers can be found in each application's `README.md` file (in their respective directories). 

## Related Code
  - Dockerfile for AMD GPU environment setup
    https://gem5.googlesource.com/public/gem5/+/refs/heads/stable/util/dockerfiles/gcn-gpu/Dockerfile
  - ResNet-50 with KFAC Pytorch
    https://github.com/gpauloski/kfac_pytorch.git
  - Public development project of the LAMMPS MD Software Package
    https://github.com/lammps/lammps
  - Accel-Sim Repo (used for PageRank)
    https://github.com/accel-sim/gpu-app-collection
