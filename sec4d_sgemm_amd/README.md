# Section 4D: SGEMM on AMD GPUs

## Application Overview and Directory Structure

Our application utilizes the SGEMM kerenl in AMD's hipBLAS library to perform matrix multiplication on two matrices containing single-precision floats. Below is a breakdown of this directory. 

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

## Adjusting Input Configurations

By default, `run-sgemm-amd.sh` performs 100 kernels of matrix multiplication on GPU 0 
with input matrices of size `24576x24576`. These parameters can be adjusted `run-sgemm-amd.sh`. Simply change the value after `NUM_KERN`, `DEVICE_ID` and/or `SIZE` in `run-sgemm-amd.sh`. 

## Running SGEMM on NVIDIA GPUs

There are two steps to build and run SGEMM on NVIDIA GPUs:
```
./build-sgemm-amd.sh
./run-sgemm-amd.sh
```
You will find two output files in `../out`. One will be `../out/sgemm-amd-*.csv` and the other will be `../out/sgemm-amd-*.txt`. The `.csv` file contains kernel runtime information and the `.txt` file contains frequency, power, and temperature measurements over the duration of the application run.