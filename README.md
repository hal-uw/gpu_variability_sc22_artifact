# Experiments Ran in "Characterizing Variability in Large-Scale, Accelerator-Rich Systems" Paper

This artifact contains code to reproduce the experiments carried out in "Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems". The repository is organized into directories per experiment. See [Experiments](#experiments) below for an overview of the contents in each directory, and [Build and Run](#build-and-run) for step-wise instructions to reproduce our experiments. 

## Table of Contents

- [Experiments](#experiments)
- [Build and Run](#build-and-run)
- [Related Code](#related-code)

## Experiments

There were 6 experiments we ran in our paper: 
- **SGEMM on NVIDIA V100s**: an application we wrote that utilizes NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats 
- **SGEMM on AMD MI60s**: we also wrote sgemm with AMD's rocBLAS library
- **ResNet-50 on NVIDIA V100s**: a stable CNN application used widely in the HPC community as a benchmark
- **BERT on NVIDIA V100s**: popular transformer-based model used in NLP, and part of the MLPerf suite
- **LAMMPS on NVIDIA V100s**: a molecular dynamics application widely used for scalable science experiments 
- **PageRank on NVIDIA V100s**: a graph analytics benchmark that uses spare matrix vector multiplication

For each experiment, there is a corresponding directory in this repository (e.g., SGEMM on NVIDIA: `sec4bc_sgemm_nvidia`). The `sec` prefixes in each directory name correspond to the sections in our paper.

## Build and Run
To run each of our applications, we provide scripts that pull container images in each application directory. We used Singularity version 3.7.2-4.el7a (stable version as of 07/26/2022) for testing all scripts. These containers install all dependencies and compile library code into respective application binaries. Directions to run each application using Singularity can be found in each application's `README.md` file (in their respective directories). 

### Steps to reproduce experiments
1. Login to a **compute node** with a GPU. All the steps that follow should be run on the compute node.
2. Clone this artifact repository on the compute node:
    ```
    git clone --recurse-submodules https://github.com/hal-uw/gpu_variability_sc22_artifact.git
    cd gpu_variability_sc22_artifact/
    ```
3.  Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity.   Please refer to this [gist](https://gist.github.com/shivaram/80f4d8a48fb4cdd52348c37508054cee) if you need to install Singularity.  
  Note that all steps and scripts in this appendix are tested with Singularity v3.7.2-4.el7a. 
4. For NVIDIA GPUs, check if the command `nvidia-smi` is printing out information associated with the attached GPUs. If not, you will need to install relevant [NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) for your Linux distribution and [Nvidia container toolkit](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html#enabling-the-docker-repository-and-installing-the-nvidia-container-toolkit). Please note that these requisites are generally pre-installed on GPU compute nodes in supercomputing systems, and likely don't require manual installation.
5. If the compute node has an NVIDIA GPU, then you can run experiments corresponding to the following directories - `sec4bc_sgemm_nvidia`, `sec5a_resnet`, `sec45b_lammps` and `sec5c_pagerank`. If the node has an AMD GPU, cd to `sec4d_sgemm_amd` for running the SGEMM application. 
    ```
    cd gpu_variability_sc22_artifact/<experiment_specific_directory>
    ```
6. Navigate to the corresponding `README.md` file inside the experiment-specific directory, and follow the steps mentioned in [Pull Container Image and Run the Application] section of each README. 

## Related Code
  - ResNet-50 with KFAC Pytorch
    https://github.com/gpauloski/kfac_pytorch.git
  - PyTorch Dataloader for ImageNet V2
    https://github.com/modestyachts/ImageNetV2_pytorch
  - Public development project of the LAMMPS MD Software Package
    https://github.com/lammps/lammps
  - Accel-Sim Benchmarks Repo (used for PageRank)
    https://github.com/accel-sim/gpu-app-collection
