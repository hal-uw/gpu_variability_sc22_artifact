# ResNet-50 on NVIDIA GPUs

## Application Overview and Directory Structure
We ran the training phase of ResNet-50 CNN. We chose the 50-layer version because it is a stable, commonly used benchmark in the HPC community. Our training set was 1.2 million images from ImageNet and our batch size was 64. We define one training run as 500 iterations. Note that we did not complete training on the entire training set; 500 training iterations was sufficiently long to collect profiling data while training was stable. Our training ran across four GPUs on one node. Because we only use one node, we do not need to use any `mpi` commands. 
For running ResNet, please see sections [Prerequisites](#prerequisites) and [Running ResNet-50 on NVIDIA GPUs](#running-the-application).

Below is a breakdown of this directory:

```
├── cnn_utils/: Directory containing utility files for PyTorch ResNet implementation
├── torch_imagenet_resnet.py: Main Python file to launch ResNet
├── utils.py: Utility functions imported into torch_imagenet_resnet.py
├── README.md: This file, containing instructions and information
├── dataloader.sh: Script to build and install ImagenetV2 for dataset generation
├── resnet-singularity-single.sh: Top-level shell script for single GPU implementation
├── resnet-singularity-multi.sh: Top-level shell script for multi-GPU implementation
├── run-resnet-single.sh: Script to run ResNet-50 on a single NVIDIA GPU
├── run-resnet-multi.sh: Script to run ResNet-50 on multiple NVIDIA GPUs using parallel processing
├── slurm-run-resnet-single.slurm: SLURM script for running single GPU implementation
├── slurm-run-resnet-multi.slurm: SLURM script for running multi-GPU implementation
├── generate_slurms_single.py: Script to generate SLURM jobs for single GPU implementation
├── generate_slurms_multi.py: Script to generate SLURM jobs for multi-GPU implementation
├── extract_data_single.py: Script to aggregate data from single GPU runs
├── extract_data_multi.py: Script to aggregate data from multi-GPU runs
```

Note: `resnet-singularity.sh` and `run-resnet.sh` now have single GPU and multi-GPU implementations.

## Adjusting Input Configurations
To adjust configuration parameters, update run-resnet-single.sh or 'run-resnet-multi.sh. Specifically, update `NGPUS` and `NNODES` to adjust the number of gpus and/or nodes, respectively. Update line 19 to adjust the batch size. The default number of gpus is 4, number of nodes is 1, and batch size is 64. Finally, to adjust the number of training iterations, change line 38 in `sec5a_resnet/cnn_utils/engine.py`.

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (if not, please refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Compilation and launch scripts have been tested with Volta V100 GPUs (arch_70, compute_70) and Quadro RTX 5000 (arch_75, compute_75), but the scripts also work for any other NVIDIA GPU.
* Machine's underlying architecture must be either `x86_64` or `amd64` (To check, run `uname -m`)
* The PyTorch NGC container we use requires the host system to have the following installed: [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html), [NVIDIA GPU Drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Please note that these requisites are generally pre-installed/pre-built on GPU compute nodes in supercomputing systems, and likely don't require manual installation.

## Run the Application
For running ResNet-50, we use a PyTorch container provided by Nvidia GPU Cloud (NGC), pulled using Singularity. The container image works only if the underlying architecture is either `x86_64` or `amd64`. Steps to run ResNet-50 using a Singularity container are given below: 

(1) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. 

(2) Run the top-level script `resnet-singularity.sh`
```
./resnet-singularity.sh
```

This script first pulls the PyTorch NGC container image and generates the Imagenet V2 Dataset using a [PyTorch dataloader](https://github.com/modestyachts/ImageNetV2_pytorch), which is added as a dependency of this repository. It finally runs 500 iterations of training and profiles the kernel information as well as other metrics using _nvprof_.  

There will be 4 csv files and 1 txt file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. They will be present in the current working directory:
  - `resnet_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. There will be one csv file per GPU (e.g., if you trained on 4 GPUs, there will be 4 csv files).
  - `resnet_iterdur_*.txt`: contains iteration durations. Iteration durations are directly printed from line 75 in `sec5a_resnet/cnn_utils/engine.py`. Only one text file. 

PLease refer to [Troubleshooting](#troubleshooting) if you see any errors while running the above steps. 

## Build and Run Without a Container Image
There are 3 steps to take to run ResNet-50 using shell scripts:
1. Setup your environment. For our purposes, we setup a Conda environment and separately installed Pytorch 1.9.0. Note that the steps for creating a Conda environment will change depending on the machine and software stack available. Many systems come with PyTorch Conda environments so it is recommended to clone the provided environment and use that instead.
```
$ conda create -n {ENV_NAME} python=3.8
$ conda activate {ENV_NAME}
$ conda env update --name {ENV_NAME} --file environment.yml
$ pip install -r requirements.txt
```
2. Run `dataloader.sh`. This builds and installs the ImageNetV2_Pytorch library and allows dataset loading functions to be used by the ResNet application.
3. Run `run-resnet-single.sh` or `run-resnet-multi.sh` after making the changes to the files and directories below
   (Note: `run-resnet-single.sh`is used for running ResNet-50 on a single GPU and `run-resnet-multi.sh`is used for running ResNet-50 on multiple GPUs in parallel.)

## Changes to Files and Directories

### Create Data Directories

1. **Create Data Directories**
   - For Single GPU: Create a new directory in your `sec5a_resnet` folder called `data`.
   - For Multi GPU: Create a new directory in your `sec5a_resnet` folder called `data_multi`.

### Configure SLURM Scripts

2. **Configure SLURM Scripts**
   - For Single GPU: Edit `slurm-run-resnet-single.slurm` line 9 to your email address for job updates.
   - For Multi GPU: Edit `slurm-run-resnet-multi.slurm` line 9 to your email address for job updates.

### Pull Docker Files

3. **Pull Docker Files**
   - For Single GPU: Open `resnet-singularity-single.sh` and uncomment line 9 to pull the required Docker files for training data. Once you run a job, you can comment the line out again.
   - For Multi GPU: Open `resnet-singularity-multi.sh` and uncomment line 9 to pull the required Docker files for training data. Once you run a job, you can comment the line out again.

### Edit Training Script Paths

4. **Edit Training Script Paths**
   - Edit `torch_imagenet_resnet.py` lines 33 and 35 with the correct paths to your training data directory and validation directory. This step is necessary for both Single GPU and Multi GPU setups.

### Configure SLURM Generation Scripts

5. **Configure SLURM Generation Scripts**
   - For Single GPU: Edit `generate_slurms_single.py` line 12 to your email and make the `exclude_list` on line 42 an empty list if running for the first time. Later on, enter the cabinet number and the node that you already have data on (e.g., `c302-001` means cabinet 302, Node 1).
   - For Multi GPU: Edit `generate_slurms_multi.py` line 12 to your email and make the `exclude_list` on line 44 an empty list if running for the first time. Later on, enter the cabinet number and the node that you already have data on (e.g., `c302-001` means cabinet 302, Node 1).

### Submit Job

6. **Submit Job**
   - For Single GPU: To submit the job to an infrastructure using an `sbatch` queue system, use the command:
     ```sh
     sbatch slurm-run-resnet-single.slurm
     ```
   - For Multi GPU: To submit the job to an infrastructure using an `sbatch` queue system, use the command:
     ```sh
     sbatch slurm-run-resnet-multi.slurm
     ```
   - These scripts are used to test if everything’s working okay on one node. Once confirmed, scale out the experiment to all 84 nodes by running the respective `generate_slurms` script:
     - For Single GPU: `generate_slurms_single.py`
     - For Multi GPU: `generate_slurms_multi.py`

### Check Job Status

7. **Check Job Status**
   - To check jobs in the queue, use the command:
     ```sh
     squeue -u yourusernamehere
     ```

## Aggregating Data

8. **Aggregate Data**
   - For Single GPU: Edit `extract_data_single.py` line 95 to the location of your `data` folder created in step 1.
   - For Multi GPU: Edit `extract_data_multi.py` line 95 to the location of your `data_multi` folder created in step 1.
   - If running these files causes an error with permissions, use the command:
     ```sh
     chmod -R 0777 data/  # For Single GPU
     chmod -R 0777 data_multi/  # For Multi GPU
     ```
   - These scripts will output two files each:
     - For Single GPU: `all_data.csv` and `aggregated_data.csv`
     - For Multi GPU: `all_data_multi.csv` and `aggregated_data_multi.csv`
     - `all_data.csv` and `all_data_multi.csv`: CSV files with all training times for each GPU, as each GPU trains the data 5 times.
     - `aggregated_data.csv` and `aggregated_data_multi.csv`: CSV files with the average training time of these 5 iterations summarized into one entry.

    ## Troubleshooting
* __python3: can't open file 'setup.py': [Errno 2] No such file or directory__
  
  Run the following command:
  ```
  git submodule update --init --recursive
  ```
  We use an [ImageNetV2_PyTorch](https://github.com/modestyachts/ImageNetV2_pytorch) dataset that is added as a git submodule within `utils/dataloader`. The above command should update `dataloader` directory with the `setup.py` and other files required for generating the ImageNet V2 dataset. 
