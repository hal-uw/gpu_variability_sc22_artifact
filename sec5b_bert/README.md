# Section 5E: BERT Pretraining on NVIDIA GPUs

## Application Overview and Directory Structure
We ran the pre-training phase of BERT-Large Uncased with 1024 hidden layers. We only consider the first phase of pre-training with sequences of length 128. Our training set is the Wikipedia EN corpus. Our training ran across four GPUs on one node with a single run as 250 iterations. Because we only use one node, we do not need to use any `mpi` commands. 
For running BERT, please see sections [Prerequisites](#prerequisites) and [Pull Container Image and Run the Application](#pull-container-image-and-run-the-application).

Below is a breakdown of this directory. 
```
├── config: a directory containing configuration details regarding locating the dataset and other parameters
├── src: a directory containing source files used in the pytorch BERT implementation
├── utils: a directory containing utility files used in downloading and encoding the dataset
├── run_pretraining.py: main python file called to run BERT pretraining
├── README.md: contains BERT Pretraining specific instructions on running the application and adjusting input configurations
├── bert-singularity.sh: Top-level script that pulls a PyTorch container image using Singularity, installs dependencies and runs BERT
```

## Adjusting Input Configurations
To adjust configuration parameters, update `scripts/run_pretraining_lamb.sh`. Specifically, update `PROC_PER_NODE` (line 11, default 4) and `NODES` (line 12, default 1) to adjust the number of gpus and/or nodes to run on, respectively. We provide a reduced size dataset within this repository. However, if you wish to run with a different dataset, you will need to update `config/bert_pretraining_phase1_config.json` so the vocab file points correctly to your given data file as well as `scripts/run_pretraining_lamb.sh` to point to the different data directory.

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (if not, please refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* The scripts provided here have been tested with V100 GPUs (Volta70) and Quadro RTX 5000 (Turing75), but the scripts also work for any other NVIDIA GPU.
* Machine's underlying architecture must be either `x86_64` or `amd64` (To check, run `uname -m`)
* The base image we use ([NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch))requires the host system to have the following installed: [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html), [NVIDIA GPU Drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Please note that these requisites are generally pre-installed on GPU compute nodes in supercomputing systems, and likely don't require manual installation.

## Pull Container Image and Run the Application
For running BERT, we use a PyTorch container provided by Nvidia GPU Cloud (NGC), pulled using Singularity. The container image works only if the underlying architecture is either `x86_64` or `amd64`. Steps to run ResNet-50 using a Singularity container are given below: 

(1) Ensure that Singularity is installed/loaded on the compute node. Compute nodes on most HPC clusters have singularity pre-installed as a module, which needs to be loaded using cluster-specific commands. For instance, on any Texas Advanced Computing Center (TACC) cluster, `module load tacc-singularity` loads the latest stable version of Singularity. 

(2) Run the top-level script `resnet-singularity.sh`
```
$ ./bert-singularity.sh
```
This script pulls the PyTorch image and installs dependencies such as `boto3`, `h5py` and `tokenizers`, before running 250 iterations of BERT training and profiling relevant metrics using _nvprof_. You will find the following output files in the current working directory:
  - `bert_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. During the profiling process, `nvprof` generates multiple such CSV files, not all of which have useful information. The number of CSV files with useful information corresponds to the number of GPUs that BERT is being run on.  


## Build and Run Without Container Image
There are the steps to take to run BERT Pretraining using shell scripts:
1. Setup your environment. During our testing, we setup a Conda environment and separately installed Pytorch 1.9.0. Note that the steps for creating a Conda environment will change depending on the machine and software stack available. Many systems come with PyTorch Conda environments so it is recommended to clone the provided environment and use that instead.
```
$ conda create -n {ENV_NAME} python=3.8
$ conda activate {ENV_NAME}
$ conda env update --name {ENV_NAME} --file environment.yml
$ pip install -r requirements.txt
```

2. Install NVIDIA APEX. Note this step requires `nvcc` and may fail if done on systems without a GPU (i.e. you may need to install on a compute node).
```
$ cd utils/apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
3. If you want to run with the reduced-size dataset that we provide by default, skip to **step 6**. 
If you want to run with some other dataset or want to download the dataset and configuration files, run `scripts/create-datasets.sh` with the appropriate flags which downloads and encodes the Wikipedia English corpus (example below). This may take a couple of hours.
```
$ ./scripts/create_datasets.sh --output <DATA_DIR> --nproc 8 --download --no-books --format --encode --encode-type bert
```
4. Update line 6 in `scripts/run-pretraining-lamb.sh' to provide the directory where the encoded training dataset is located on your machine. (e.g. <DATA_DIR>/encoded/sequences_lowercase_max_seq_len_128_next_seq_task_true)
5. Update line 17 in `config/bert_large_uncased_config.json` to provide the correct path to the vocab.txt that was also downloaded as part of Step 1 (e.g. <DATA_DIR>/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt).
6. Run `chmod u+x scripts/run-pretraining-lamb.sh scripts/launch-pretraining.sh`.
7. Run `scripts/run-pretraining-lamb.sh`.

You will find a few output files in `in this directory`:
  - `bert_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. During the profiling process, `nvprof` generates multiple such CSV files, not all of which have useful information. The number of CSV files with useful information corresponds to the number of GPUs that BERT is being run on.  
