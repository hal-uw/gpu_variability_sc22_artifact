# Section 4B & 4C: SGEMM on NVIDIA GPUs

## Application Overview and Directory Structure

Our application utilizes SGEMM kernels in NVIDIA's cuBLAS library to perform matrix multiplication on two matrices containing single-precision floats. Below is a breakdown of this directory.
```
├── gen_data.cpp: generates two input matrices of a size the user specifies
├── gputimer.h: 
├── Makefile: make binaries for `gen_data.cpp` and `sgemm.cu`
├── README.md: contains SGEMM specific instructions on running the application and configuring input size
├── sgemm.cu: main application that uses matrices generated from gen-data.cpp as inputs
├── build-sgemm-nvidia.sh: builds sgemm application for NVIDIA GPUs
├── run-sgemm-nvidia.sh: runs sgemm application on NVIDIA GPUs
```

## Adjusting Input Configurations

By default, `run-sgemm-nvidia.sh` performs 100 kernels of matrix multiplication on GPU 0 
with input matrices of size `25536x25536`. These parameters can be adjusted `run-sgemm-nvidia.sh`. Simply change the value after `NUM_KERN`, `DEVICE_ID` and/or `SIZE` in `run-sgemm-nvidia.sh`. 

## Running SGEMM on NVIDIA GPUs

There are four steps to build and run SGEMM on NVIDIA GPUs:
```
chmod u+x ./build-sgemm-nvidia.sh
chmod u+x ./run-sgemm-nvidia.sh
./build-sgemm-nvidia.sh
./run-sgemm-nvidia.sh
```
You will find the outputs from the `nvprof` profiler in `../out/sgemm-nvidia-*.csv`. 