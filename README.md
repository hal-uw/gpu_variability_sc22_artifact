# Characterizing Variability in Large-Scale, Accelerator-Rich Systems

This artifact contains code to reproduce the experiments carried out in "Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems". 

## Table of Contents

- [Experiments](#experiments)
- [Install and Build](#install-build)
- [Usage](#usage)
- [Related Code](#related)
- [Citation](#citation)

## Experiments

### Section 4B & 4C: SGEMM on NVIDIA GPUs
We wrote a benchmark that utilizes SGEMM kernels in NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats. Below is a breakdown of the `sec4bc_sgemm_nvidia` directory. 
```
├── sec4bc_sgemm_nvidia
│   ├── gen_data.cpp: generates two input matrices of a size the user specifies
│   ├── gputimer.h: 
│   ├── Makefile: make binaries for `gen_data.cpp` and `sgemm.cu`
│   ├── README.md: contains SGEMM specific instructions on running the application and configuring input size
│   ├── sgemm.cu: main application that uses matrices generated from gen-data.cpp as inputs
│   ├── build-sgemm-nvidia.sh: builds sgemm application for NVIDIA GPUs
│   ├── run-sgemm-nvidia.sh: runs sgemm application on NVIDIA GPUs
```

Both `build-sgemm-nvidia.sh` and `run-sgemm-nvidia.sh` are called by `build-all.sh` and `run-all.sh`, respectively (See [Install and Build](#install-and-build)). By default, `build-all.sh` builds two input matrices of size `25536x25536` and runs 100 kernels of SGEMM on one GPU (GPU or device ID 0). The `nvprof` profiled output can be found in `out/sgemm_nvidia_*.csv`. To make changes to the default input size, number of kernels run, or which GPU the kernels are run on, edit [`sec4bc_sgemm_nvidia/run-sgemm-nvidia.sh`](). More instructions can be found in [SGEMM NVIDIA README](/sec4bc_sgemm_nvidia/README.md).

### Section 4D: SGEMM on AMD GPUs
In addition to writing an SGEMM application for NVIDIA GPUs, we also wrote an equivalent version for AMD GPUs. We use the SGEMM kernel from AMD's hipBLAS library. Below is a breakdown of the `sec4d_sgemm_amd` directory.
```
├── sec4d_sgemm_amd
│   ├── gen_data.cpp: generates two input matrices of a size the user specifies
│   ├── gputimer.hip.h: 
│   ├── Makefile: make binaries for `gen_data.cpp` and `sgemm.cu`
│   ├── README.md: contains SGEMM specific instructions on running the application and configuring input size
│   ├── sgemm_rocblas.hip.cpp: main application that uses matrices generated from gen-data.cpp as inputs
│   ├── build-sgemm-amd.sh: builds sgemm application for AMD GPUs
│   ├── run-sgemm-amd.sh: runs sgemm application on AMD GPUs
```
Both `build-sgemm-amd.sh` and `run-sgemm-amd.sh` are called by `build-all.sh` and `run-all.sh`, respectively (See [Install and Build](#install-and-build)). By default, `build-all.sh` builds two input matrices of size `24576x24576` and runs 100 kernels of SGEMM on one GPU (GPU or device ID 0). Unlike with NVIDIA GPUs, two profilers are employed to collect data: `rocprof` and `rocm-smi`. Hence, two different output files are generated: `out/sgemm_amd_*.csv`, which contains kernel runtime information, and `out/sgemm_amd_*.txt`, which contains GPU SM frequency, power, and temperature measurements collected over the duration of the application run. To make changes to the default input size, number of kernels run, or which GPU the kernels are run on, edit [`sec4d_sgemm_amd/run-sgemm-amd.sh`](). More instructions can be found in [SGEMM AMD README](/sec4d_sgemm_amd/README.md). 

### Section 5A: ResNet on NVIDIA GPUs
We ran the training phase of ResNet-50 CNN. We chose the 50-layer version because it is a stable, commonly used benchmark in the HPC community. Our training set was 1.2 million images from ImageNet and our batch size was 64. We define one training run as 500 iterations. Note that we did not complete training on the entire training set; 500 training iterations was sufficiently long to collect profiling data while training was stable. Our training ran across four GPUs on one node. Because we only use one node, we do not need to use any `mpi` commands. Below is a breakdown of the `sec5a_resnet` directory. 

```
├── sec5a_resnet
│   ├── cnn_utils: a directory containing utility files used in the pytorch resnet implementation
│   ├── torch_imagenet_resnet.py: main python file called to launch ResNet
│   ├── utils.py: utility functions imported into torch_imagenet_resnet.py
│   ├── README.md: contains ResNet-50 specific instructions on running the application and adjusting input configurations
│   ├── run-resnet.sh: runs ResNet-50 on NVIDIA GPUs
```
Note that there is not build script (e.g., `build-resnet.sh`), as the application is written in Python. The `run-resnet.sh` script is called by `run-all.sh` (See [Install and Build](#install-and-build)). To adjust input configuration parameters (e.g., number of iterations, batch size), update run-resnet.sh. There are two output files for our ResNet-50 application. Kernel duration, GPU SM frequency, power, and temperature can all be found in `out/resnet_*.csv`. We also print out iteration duration directly from the Python application and pipe it to `out/resnet_iterdur_*.csv`. More information can be found in [the ResNet-50 README](sec5a_resnet/README.md).

### Section 5B: LAMMPS on NVIDIA GPUs
We ran REAXC setting within LAMMPS, with inputs sized for max GPU occupancy, while staying within memory limits for a single GPU application. The `src` directory has all packages and `Makefile`s for compiling various LAMMPS binaries, while `reax_benchmark` directory contains scripts for building and running LAMMPS. These scripts are called by `build-all.sh` and `run-all.sh` (See Install and Build). By default, `run-all.sh` runs 2 single-GPU LAMMPS jobs on GPU 0 (device ID 0) and stores the nvprof profiled output to `out/lammps-*.csv`. To make changes to the default input configuration, number of runs, output file etc, edit `sec5b_lammps/lammps_17Jan18/reax_benchmark/run-lammps.sh`

### Section 5C: PageRank on NVIDIA GPUs
We used the PageRank Sparse-matrix Vector Multiplication (SPMV) program available at https://github.com/accel-sim/gpu-app-collection. PageRank is also run as a single-GPU job, with an undirected input graph `rajat30.mtx` sourced from the SuiteSparse Matrix Collection (https://sparse.tamu.edu/Rajat/rajat30). This graph is present within the artifact at `sec5c_pagerank/gpu-app-collection/data_dirs/pannotia/pagerank_spmv/data`

## Install and Build

@Rutwik - work in progress 
```
```

## Usage

## Related Code
    - Public development project of the LAMMPS MD Software Package
      https://github.com/lammps/lammps
    - Accel-Sim Repo (used for PageRank)
      https://github.com/accel-sim/gpu-app-collection

## Citation
