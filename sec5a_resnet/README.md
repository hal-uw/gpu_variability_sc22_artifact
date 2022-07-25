# Section 5A: ResNet-50 on NVIDIA GPUs

## Application Overview and Directory Structure
We ran the training phase of ResNet-50 CNN. We chose the 50-layer version because it is a stable, commonly used benchmark in the HPC community. Our training set was 1.2 million images from ImageNet and our batch size was 64. We define one training run as 500 iterations. Note that we did not complete training on the entire training set; 500 training iterations was sufficiently long to collect profiling data while training was stable. Our training ran across four GPUs on one node. Because we only use one node, we do not need to use any `mpi` commands. 
For running ResNet, please see sections [Prerequisites](#prerequisites) and [Running ResNet-50 on NVIDIA GPUs](#run-the-application).

Below is a breakdown of this directory. 
```
├── cnn_utils: a directory containing utility files used in the pytorch resnet implementation
├── torch_imagenet_resnet.py: main python file called to launch ResNet
├── utils.py: utility functions imported into torch_imagenet_resnet.py
├── README.md: contains ResNet-50 specific instructions on running the application and adjusting input configurations
├── dataloader.sh: script that builds and installs ImagenetV2 for generating the dataset
├── resnet-singularity.sh: top-level shell script that pulls a container image, compiles binary and related packages and runs ResNet-50
├── run-resnet.sh: runs ResNet-50 on NVIDIA GPUs by calling torch_imagenet_resnet.py with relevant args
```

## Adjusting Input Configurations
To adjust configuration parameters, update `run-resnet.sh`. Specifically, update `NGPUS` and `NNODES` to adjust the number of gpus and/or nodes, respectively. Update line 19 to adjust the batch size. The default number of gpus is 4, number of nodes is 1, and batch size is 64. Finally, to adjust the number of training iterations, change line 38 in `sec5a_resnet/cnn_utils/engine.py`.

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (if not, please refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Compilation and launch scripts have been tested with Volta V100 GPUs (arch_70, compute_70) and Quadro RTX 5000 (arch_75, compute_75), but the scripts also work for any other NVIDIA GPU.
* Machine's underlying architecture must be either `x86_64` or `amd64` (To check, run `uname -m`)
* The PyTorch NGC container we use requires the host system to have the following installed: [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html), [NVIDIA GPU Drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Please note that these requisites are generally pre-installed/pre-built on GPU compute nodes in supercomputing systems, and likely don't require manual installation.

## Run the Application
For running ResNet-50, we use a PyTorch container provided by Nvidia GPU Cloud (NGC), pulled using Singularity. The container image works only if the underlying architecture is either `x86_64` or `amd64`. Steps to run ResNet-50 using a Singularity container are given below: 

(1) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. Note that these steps and scripts are tested with Singularity v3.7.2-4.el7a. 

(2) Run the top-level script `resnet-singularity.sh`
```
./resnet-singularity.sh
```

This script first pulls the PyTorch NGC container image and generates the Imagenet V2 Dataset using a [PyTorch dataloader](https://github.com/modestyachts/ImageNetV2_pytorch), which is added as a dependency of this repository. It finally runs 500 iterations of training and profiles the kernel information as well as other metrics using _nvprof_.  

There will be 4 csv files and 1 txt file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. They will be present in the current working directory:
  - `resnet_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. There will be one csv file per GPU (e.g., if you trained on 4 GPUs, there will be 4 csv files).
  - `resnet_iterdur_*.txt`: contains iteration durations. Iteration durations are directly printed from line 75 in `sec5a_resnet/cnn_utils/engine.py`. Only one text file. 

## Build and Run Without Docker
There are 4 steps to take to run ResNet-50 using shell scripts:
1. Setup your environment. For our purposes, we setup a Conda environment and separately installed Pytorch 1.9.0. Note that the steps for creating a Conda environment will change depending on the machine and software stack available. Many systems come with PyTorch Conda environments so it is recommended to clone the provided environment and use that instead.
```
$ conda create -n {ENV_NAME} python=3.8
$ conda activate {ENV_NAME}
$ conda env update --name {ENV_NAME} --file environment.yml
$ pip install -r requirements.txt
```
2. Run `dataloader.sh`. This builds and installs the ImageNetV2_Pytorch library and allows dataset loading dunctions to be used by the ResNet application. 
4. Run `run-resnet.sh`. 

You will find a few output files in the working directory:
  - `resnet_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. There will be one csv file per GPU (e.g., if you trained on 4 GPUs, there will be 4 csv files).
  - `resnet_iterdur_*.txt`: contains iteration durations. Iteration durations are directly printed from line 75 in `sec5a_resnet/cnn_utils/engine.py`. Only one text file. 
