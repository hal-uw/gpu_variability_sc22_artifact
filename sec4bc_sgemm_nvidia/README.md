# Section 4B & 4C: SGEMM on NVIDIA GPUs

## Application Overview and Directory Structure

Our application utilizes SGEMM kernels in NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats. We ran it as a single-GPU application using NVIDIA V100 GPUs and allowed the application to run to completion. 

For compiling and launching SGEMM on NVIDIA GPUs, please see sections [Prerequisites](#prerequisites) and [Pull Container Image and Run the Application](#pull-container-image-and-run-the-application). Below is a breakdown of this directory.
```
├── gen_data.cpp: generates two input matrices of a size the user specifies
├── gputimer.h: header file to create manual timer for CUDA calls
├── Makefile: make binaries for `gen_data.cpp` and `sgemm.cu`
├── README.md: contains SGEMM specific instructions on running the application and configuring input size
├── sgemm.cu: main application that uses matrices generated from gen-data.cpp as inputs
├── build-sgemm-nvidia.sh: script used with Singularity container to build sgemm binary (can be used to run without container))
├── run-sgemm-nvidia.sh: script used with Singularity container to run sgemm (can be used to run without docker)
├── sgemm-singularity.sh: Top-level shell script that pulls container, compiles binary and related packages and runs SGEMM
```

## Adjusting Input Configurations

By default, `run-sgemm-nvidia.sh` performs 100 kernels of matrix multiplication on GPU 0 
with input matrices of size `25536x25536`. These parameters can be adjusted in `run-sgemm-nvidia.sh`. Simply change the value after `NUM_KERN`, `DEVICE_ID` and/or `SIZE` in `run-sgemm-nvidia.sh`. 

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (if not, please refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Compilation and launch scripts assume Volta Class GPU(s) (arch_70, compute_70) are available on the compute node, but they work for any other NVIDIA GPU. 
* If your GPU is not a Volta, edit line 4 of `Makefile` with the following compute_/arch_/SM_ tokens as applicable: 


| NVIDIA Architecture        | Cards                                   | Supported Sm and Gencode Variations |
|:---------------------------|:----------------------------------------|:------------------------------------|
| Fermi (CUDA 3.2 until 8)   | GeForce 400, 500, 600, GT-630           | `SM_20` `compute_30`                |
| Kepler(CUDA 5 until 10)    | GeForce 700, GT-730                     | `SM_30` `compute_30`                |
|                            | Tesla K40                               | `SM_35` `compute_35`                |
|                            | Tesla K80                               | `SM_37` `compute_37`                |
| Maxwell (CUDA 6 until 11)  | Tesla/Quadro M                          | `SM_50` `compute_50`                |
|                            | Quadro M6000, GeForce 900,              | `SM_52` `compute_52`                |
|                            | GTX-970, GTX-980, GTX Titan-X           |                                     |
|                            | Jetson TX1/Tegra X1,                    | `SM_53` `compute_53`                |
|                            | Drive CX/PX, Jetson Nano                |                                     |
| Pascal (>= CUDA 8)         | Quadro GP100, Tesla P100, DGX-1         | `SM_60` `compute_60`                |
|                            | GTX 1080/1070/1060/1050/1030/1010       | `SM_61` `compute_61`                |
|                            | Titan Xp, Tesla P40, Tesla P4           |                                     |
|                            | iGPU on Drive PX2, Tegra(Jetson) TX2    | `SM_62` `compute_62`                |
| Volta (>= CUDA 9)          | **Tesla V100**, DGX-1, GTX1180,         | `SM_70` `compute_70`                |
|                            | Titan V, Quadro GV100                   |                                     |
|                            | Jetson AGX Xavier, Drive AGX Pegasus    | `SM_72` `compute_72`                |
|                            | Xavier NX                               |                                     |
| Turing (>= CUDA 10)        | GTX 1660, RTX 20X0 (X=6/7/8), Titan RTX | `SM_75` `compute_75`                |
|                            | Quadro RTX 4000/5000/6000/8000,         |                                     |
|                            | Tesla T4                                |                                     |
| Ampere (>= CUDA 11.1)      | A100, GA100, DGX-A100                   | `SM_80` `compute_80`                |
|                            | GA10X cards, RTX 30X0 (X=5/6/7/8/9)     | `SM_86` `compute_86`                |

## Pull Container Image and Run the Application 
We use pre-existing Docker images from nvidia/cuda, pulled using Singularity. Steps to build and run SGEMM using a Singularity container:

(1) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. 
Note that these steps and scripts are tested with Singularity v3.7.2-4.el7a. 

(2) Run the top-level script `sgemm-singularity.sh`. This script pulls the relevant container and runs all compilation and application execution steps to return output. 
```
./sgemm-singularity.sh
```

There will be one csv file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. This file will be present in the current directory. The name of the file is of the format `sgemm_nvidia_25536_100_<UUID>_<DEVICE_ID>_<TIMESTAMP>.csv`, where `25536` is the input matrix size, `100` refers to the number of matrix multiplication kernels, `UUID` is the unique ID assigned to the GPU being run on, `DEVICE_ID` is the device ID of the GPU (default 0) and `TIMESTAMP` records the time at which the run started.

## Build and Run Without Docker
There are four steps to build and run SGEMM on NVIDIA GPUs:
```
chmod u+x ./build-sgemm-nvidia.sh
chmod u+x ./run-sgemm-nvidia.sh
./build-sgemm-nvidia.sh
./run-sgemm-nvidia.sh
```
You will find the output csv file from the `nvprof` profiler directly in this directory. 