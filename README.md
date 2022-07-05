# Experiments Ran in "Characterizing Variability in Large-Scale, Accelerator-Rich Systems" Paper

This artifact contains code to reproduce the experiments carried out in "Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems". The repository is organized into directories per experiment. See [Experiments](#experiments) below for an overview of the contents in each directory, and [Build and Run](#build-and-run) for step-wise instructions to reproduce our experiments. 

## Table of Contents

- [Experiments](#experiments)
- [Build and Run](#build-and-run)
- [Related Code](#related-code)

## Experiments

There were 5 experiments we ran in our paper: 
- **SGEMM on NVIDIA V100s**: an application we wrote that utilizes NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats 
- **SGEMM on AMD MI60s**: we also wrote sgemm with AMD's rocBLAS library
- **ResNet-50 on NVIDIA V100s**: a stable CNN application used widely in the HPC community as a benchmark
- **LAMMPS on NVIDIA V100s**: a molecular dynamics application widely used for scalable science experiments 
- **PageRank on NVIDIA V100s**: a graph analytics benchmark that uses spare matrix vector multiplication

For each experiment, there is a corresponding directory in this repository (e.g., SGEMM on NVIDIA: `sec4bc_sgemm_nvidia`). The names of each directory correspond to the sections in our paper.

## Build and Run
To run each of our applications, we provide docker containers in each application directory. We used docker version 20.10.16, build aa7e414 (stable version as of 05/26/2022). These docker containers install all dependencies and compile library code into respective application binaries. Directions to run each application using the docker containers can be found in each application's `README.md` file (in their respective directories). 

### Steps to reproduce experiments
1. Login to a **compute node** with a GPU. All the steps that follow should be run on the compute node.
2. Clone this artifact repository on the compute node:
```
git clone https://github.com/hal-uw/gpu_variability_sc22_artifact.git
```
3. Run `docker-install.sh` to install the latest stable version of the Docker engine (v20.10).
```
./docker-install.sh
``` 
4. For NVIDIA GPUs, enable the docker repository and install Nvidia container toolkit using instructions at https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html#enabling-the-docker-repository-and-installing-the-nvidia-container-toolkit. Note that this step requires that the relevant NVIDIA driver and Docker v20.10 or above is installed for your Linux distribution. 
5. If the compute node has an NVIDIA GPU, then you can run experiments corresponding to the following directories - `sec4bc_sgemm_nvidia`, `sec5a_resnet`, `sec45b_lammps` and `sec5c_pagerank`. If the node has an AMD GPU, cd to `sec4d_sgemm_amd` for running the SGEMM application. 
```
cd gpu_variability_sc22_artifact/<experiment_specific_directory>
```
6. Navigate to the corresponding README.md file inside the experiment-specific directory, and follow the steps mentioned in [Build Container Image] and [Run the Application] sections of each README. 

## Related Code
  - Dockerfile for AMD GPU docker based on publicy available docker from gem5, modified to remove gem5-specific parts
    https://gem5.googlesource.com/public/gem5/+/refs/heads/stable/util/dockerfiles/gcn-gpu/Dockerfile
  - ResNet-50 with KFAC Pytorch
    https://github.com/gpauloski/kfac_pytorch.git
  - Public development project of the LAMMPS MD Software Package
    https://github.com/lammps/lammps
  - Accel-Sim Benchmarks Repo (used for PageRank)
    https://github.com/accel-sim/gpu-app-collection
