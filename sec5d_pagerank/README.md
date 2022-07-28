# Section 5D: PageRank on NVIDIA GPUs

## Application Overview and Directory Structure
We ran PageRank SPMV on input graph _rajat30_ (https://sparse.tamu.edu/Rajat/rajat30), an undirected graph for a circuit simulation problem. We ran it as a single-GPU application using NVIDIA V100 GPUs and allowed the application to run to completion. Because we only use one node, we do not need to use any `mpi` commands. 

For compiling and launching PageRank, please see sections [Prerequisites](#prerequisites) and [Pull Container Image and Run the Application](#pull-container-image-and-run-the-application).
Below is a breakdown of this directory:

```
├── src: a directory containing Makefile and .cu files for compiling the PageRank binary
├── data_dirs: directory containing input graphs
├── fetch-input.sh: script to fetch input graph _rajat30.mtx_ from the SuiteSparse Matrix Collection
├── pagerank-singularity.sh: script to pull a container image, compile binary and related packages, and run PageRank within the container
├── build-pagerank.sh: script used by the Dockerfile to build PageRank (can be used to run without docker)
├── run-pagerank.sh: script used by the Dockerfile to run PageRank (can be used to run without docker)
├── README.md: contains PageRank specific instructions on running the application and adjusting input configurations
```

## Adjusting Input Configurations
Before running PageRank, the input graph _rajat30.mtx_ is fetched from SuiteSparse Matrix collection using `wget`. To change the input graph, update line 9 in `fetch-input.sh` to `wget` any other graph. 

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (if not, please refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Compilation and launch scripts assume one or more Volta Class GPUs (arch_70, compute_70) are available on the compute node. However, the scripts also work for other NVIDIA GPUs.
* If your GPU is not a Volta, edit `src/cuda/pannotia/pagerank_mod/Makefile` and uncomment the relevant `GENCODE` line (among lines 0-9 of `Makefile`) based on the table below:


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
|                            | Tesla T4                              |                                     |
| Ampere (>= CUDA 11.1)      | A100, GA100, DGX-A100                 | `SM_80` `compute_80`                |
|                            | GA10X cards, RTX 30X0 (X=5/6/7/8/9)   | `SM_86` `compute_86`                |

## Pull Container Image and Run the Application
We use an existing Docker image from nvidia/cuda, pulled using Singularity. Steps to build and run PageRank using a Singularity container are given below: 

(1) Run `fetch-input.sh` to fetch input graph _rajat30.mtx_.  
```
./fetch-input.sh
``` 

(2) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. 
Note that these steps and scripts are tested with Singularity v3.7.2-4.el7a. 

(3) Run the top-level script `pagerank-singularity.sh`. This script pulls the relevant container and runs all compilation and application execution steps to return output. 
```
./pagerank-singularity.sh
```

There will be one csv file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. This file will be present in the current directory. The name of the file is of the format `pagerank_<UUID>_run0_node<NODE_NUM>_GPU<DEVICE_ID>.csv`, `NODE_NUM` refers to the compute node ID, `UUID` is the unique ID assigned to the GPU being run on, and `DEVICE_ID` is the device ID of the GPU (default 0).

## Build and Run Without a Container Image
To run without a container image, just run the following shell script (make sure execute permissions are set for the same):
```
./run-wo-docker.sh
```
If there are any errors running the above, the following four steps can be run independently to build and run PageRank on NVIDIA GPUs:
```
chmod u+x ./fetch-input.sh
chmod u+x ./build-pagerank.sh
chmod u+x ./run-pagerank.sh
./fetch-input.sh
./build-pagerank.sh
./run-pagerank.sh 0 1 0
```
You will find the output csv file from the `nvprof` profiler directly in the current directory. 

