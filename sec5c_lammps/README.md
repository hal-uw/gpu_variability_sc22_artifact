# Section 5B: LAMMPS on NVIDIA GPUs

### Application Overview and Directory Structure
We used the LAMMPS tarball provided by the [Coral-2 benchmark suite](https://asc.llnl.gov/coral-2-benchmarks) which uses REAXC settings. We ran LAMMPS as a single-GPU experiment. For compiling and launching LAMMPS with default configuration settings, please see sections [Prerequisites](#prerequisites) and [Pull Container Image and Run the Application](#pull-container-image-and-run-the-application) below. Read on to customize run time options/arguments/input configuration.
Below is a breakdown of this directory:

```
├── src: contains Makefiles and code for compiling LAMMPS binary and associated packages
├── reax_benchmark: contains configuration parameters for the REAXC setting
├── lammps-singularity.sh: top-level script that pulls a container image, compiles the lammps binary and related packages and runs LAMMPS
├── build-lammps.sh: script used by the Dockerfile to build LAMMPS (can be used to run without container)
├── run-lammps.sh: script used with the base image to run LAMMPS (can be used to run without container)
├── README.md: contains LAMMPS-specific instructions on running the application and adjusting input configurations
```

## Adjusting Input Configurations
Each LAMMPS job is set-up to use an input configuration of 100 time steps and dimensional scaling factors (x,y,z) = (8,16,16). To change the value of x, y or z, edit the command line in `run-lammps.sh`. To change the value of time step, update `reax_benchmark/in.reaxc.hns`. 

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed
* Compilation and launch scripts assume one or more Volta Class GPUs (arch_70, compute_70) are available on the compute node. However, the scripts also work for non-Volta NVIDIA GPUs. 
* If your GPU is not a Volta, edit `CCFLAGS` and `KOKKOS_ARCH` options in `src/MAKE/OPTIONS/Makefile.kokkos_cuda_mpi` and `build-lammps.sh`
based on the following table: 

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
We use an existing [container image](https://hub.docker.com/r/pawsey/cuda-mpich-base/tags) for CUDA-aware MPICH applications, pulled using Singularity. The steps here are tested for Volta Class GPUs (Volta70) as well as Turing class GPUs (Turing75), but should work for other GPUs as long as the updates as mentioned in Pre-requisites are done. Steps to build and run SGEMM using a container image pulled using Singularity:

(1) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. 
Note that these steps and scripts are tested with Singularity v3.7.2-4.el7a. 

(2) Run the top-level script `lammps-singularity.sh`. This script pulls the relevant container image and runs all compilation and application execution steps to return output. 
```
./lammps-singularity.sh
```

There will be one csv file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. __This file can be found inside the `reax_benchmark` directory.__  The name of the file is of the format `lammps_<UUID>_<DEVICE_ID>_run<RUN_NUM>_<TIMESTAMP>.csv`, where `UUID` is the unique ID assigned to the GPU being run on, `DEVICE_ID` is the device ID of the GPU (default 0), `RUN_NUM` refers to a user-assigned number for the LAMMPS run (default 1) and `TIMESTAMP` records the time at which the run started.


## Build and Run Without Container Image
The following steps illustrate how to build and run LAMMPS on NVIDIA GPUs without using a container image. Please note that this assumes that CUDA is installed on the machine. 
```
sudo apt install mpich
chmod u+x ./src/build-lammps.sh
chmod u+x ./reax_benchmark/run-lammps.sh
cd src && ./build-lammps.sh
cd ../reax_benchmark/ && ./run-lammps.sh 0 1 0
```

There will be one csv file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. __This file can be found inside the `reax_benchmark` directory.__  The name of the file is of the format `lammps_<UUID>_<DEVICE_ID>_run<RUN_NUM>_<TIMESTAMP>.csv`, where `UUID` is the unique ID assigned to the GPU being run on, `DEVICE_ID` is the device ID of the GPU (default 0), `RUN_NUM` refers to a user-assigned number for the LAMMPS run (default 1) and `TIMESTAMP` records the time at which the run started.