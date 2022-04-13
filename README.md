# Characterizing Variability in Large-Scale, Accelerator-Rich Systems

This artifact contains code to reproduce the experiments carried out in "Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems". 

## Table of Contents

- [Experiments] (#experiments)
- [Install and Build] (#install-build)
- [Usage] (#usage)
- [Related Code] (#related)
- [Citation] (#citation)

## Experiments


### Section.5B. LAMMPS on NVIDIA GPUs
We ran REAXC setting within LAMMPS, with inputs sized for max GPU occupancy, while staying within memory limits for a single GPU application. The `src` directory has all packages and `Makefile`s for compiling various LAMMPS binaries, while `reax_benchmark` directory contains scripts for building and running LAMMPS. These scripts are called by `build-all.sh` and `run-all.sh` (See Install and Build). By default, `run-all.sh` runs 2 single-GPU LAMMPS jobs on GPU 0 (device ID 0) and stores the nvprof profiled output to `out/lammps-*.csv`. To make changes to the default input configuration, number of runs, output file etc, edit `sec5b_lammps/lammps_17Jan18/reax_benchmark/run-lammps.sh`

### Section.5C. PageRank on NVIDIA GPUs
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
