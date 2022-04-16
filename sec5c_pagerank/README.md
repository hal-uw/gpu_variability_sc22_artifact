# Section 5D: PageRank on NVIDIA GPUs

## Application Overview and Directory Structure
We ran PageRank SPMV on input graph _rajat30_ (https://sparse.tamu.edu/Rajat/rajat30), an undirected graph for a circuit simulation problem. We ran it as a single-GPU application using NVIDIA V100 GPUs and allowed the application to run to completion. Because we only use one node, we do not need to use any `mpi` commands. 

For compiling and launching PageRank, please see sections [Pre-Requisites](#pre-requisites), [Build Container Image](#build-container-image), and [Run the Application](#run-the-application).
Below is a breakdown of this directory:

```
├── src: a directory containing Makefile and .cu files for compiling the PageRank binary
├── data_dirs: directory containing input graphs
├── Dockerfile: docker to compile binary and related packages and create a container that can run PageRank directly
├── build-pagerank.sh: script used by the Dockerfile to build pagerank (can be used to run without docker)
├── run-pagerank.sh: script used by the Dockerfile to run pagerank (can be used to run without docker)
├── README.md: contains PageRank specific instructions on running the application and adjusting input configurations
```

## Adjusting Input Configurations
We have set-up the container configuration to retrieve _rajat30.mtx_ from SuiteSparse Matrix collection using `wget`. To change the input graph, update line 14 in `Dockerfile` to `wget` any other graph. 

## Pre-Requisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed
* Compilation and launch scripts assume a Volta Class GPU (arch_70, compute_70).
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
| Turing (>= CUDA 10)        | GTX 1660, RTX 20X0 (X=6/7/8), Titan RTX|| `SM_75` `compute_75`                |
|                            | Quadro RTX 4000/5000/6000/8000,         |                                     |
|                            | Tesla T4                              |                                     |
| Ampere (>= CUDA 11.1)      | A100, GA100, DGX-A100                 | `SM_80` `compute_80`                |
|                            | GA10X cards, RTX 30X0 (X=5/6/7/8/9)   | `SM_86` `compute_86`                |

## Build Container Image
Note that to successfully build this docker image and the necessary libraries/packages used for PageRank, you will
need sudo access on the machine you are doing this work. Otherwise, the container image will fail to build.
```
# Build container image
docker build -t pagerank_image .
```

## Run the Application
There will be one csv file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. This file will be stored in the docker container by default. To access this file, you will have to copy it using `docker cp` (shown below) to the directory of your choice (we recommend `../out/`).

```
# Run application
docker run --gpus all pagerank_image

# Move data output by profiler (nvprof) from container to local directory in this repository
docker create -ti --name dummy pagerank_image bash
<Returns Container ID c_id>
docker cp <c_id>:/sec5c/*.csv ../out/.
docker rm -f dummy
```

## Build and Run Without Docker
There are four steps to build and run PageRank on NVIDIA GPUs:
```
chmod u+x ./build-pagerank.sh
chmod u+x ./run-pagerank.sh
./build-pagerank.sh
./run-pagerank.sh
```

You will find the output csv file from the `nvprof` profiler directly in this directory. 

