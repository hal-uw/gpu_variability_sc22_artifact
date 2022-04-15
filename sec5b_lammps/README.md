# Section 5B: LAMMPS on NVIDIA GPUs


### Application Overview and Directory Structure
The public development project for LAMMPS MD simulation package is hosted on GitHub at https://github.com/lammps/lammps. We used the LAMMPS tarball provided by the Coral-2 suite (https://asc.llnl.gov/coral-2-benchmarks) which uses the REAXC setting as a benchmark. We ran LAMMPS as a single-GPU experiment. To reproduce our experiments as-is, refer to sections _Pre-Requisites_, _Build Container Image_ and _Run the application_ below. Read on to customize runtime options/arguments/input configuration.
Below is a breakdown of this directory:

```
├── sec5b_lammps
│   ├── src: contains Makefiles and code for compiling LAMMPS binary and associated packages
│   ├── reax_benchmark: contains configuration parameters for the REAXC setting
│   ├── Dockerfile: docker to compile binary and related packages from srf and create a container that can run LAMMPS directly
│   ├── build-lammps.sh: Shell script used by the Dockerfile (can be used to run without docker)
|   ├── run-lammps.sh: Shell script used by the Dockerfile (can be used to run without docker)
│   ├── README.md: contains  LAMMPs-specific instructions on running the application and adjusting input configurations
```

### Adjusting Input Configurations
The LAMMPS run is set-up to use an input configuration of 100 timesteps and (x,y,z) = (8,16,16). To change the value of x, y or z, edit the command line in `run-lammps.sh`. To change the value of timestep, update `reax_benchmark/in.reaxc.hns`. 

### Pre-Requisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed
* Compilation and launch scripts assume a Volta Class GPU (arch_70, compute_70).
* If your GPU is not a Volta, edit `CCFLAGS` and `KOKKOS_ARCH` options in `src/MAKE/OPTIONS/Makefile.kokkos_cuda_mpi` and `Dockerfile`
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
| Turing (>= CUDA 10)        | GTX 1660, RTX 20X0 (X=6/7/8), Titan RTX|| `SM_75` `compute_75`                |
|                            | Quadro RTX 4000/5000/6000/8000,         |                                     |
|                            | Tesla T4                              |                                     |
| Ampere (>= CUDA 11.1)      | A100, GA100, DGX-A100                 | `SM_80` `compute_80`                |
|                            | GA10X cards, RTX 30X0 (X=5/6/7/8/9)   | `SM_86` `compute_86`                |

### Build Container Image
```
# Build container image
docker build -t lammps_img .
```

### Run the application
```
# Run application
docker run --gpus all lammps_img
# Prints the profiling log name <csv_name>, move to local storage
docker cp lammps_img:<csv name> .
```
