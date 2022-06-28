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
├── Dockerfile: docker to run ResNet-50 directly
├── run-resnet.sh: runs ResNet-50 on NVIDIA GPUs (UPDATE BEFORE RUNNING)
```

## Adjusting Input Configurations
To adjust configuration parameters, update `run-resnet.sh`. Specifically, update `NGPUS` and `NNODES` to adjust the number of gpus and/or nodes, respectively. Update line 19 to adjust the batch size. The default number of gpus is 4, number of nodes is 1, and batch size is 64. Finally, to adjust the number of training iterations, change line 38 in `sec5a_resnet/cnn_utils/engine.py`.

## Prerequisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed
* Compilation and launch scripts assume one or more Volta Class GPU (arch_70, compute_70) are available on the compute node, but the scripts also work for any other NVIDIA GPU.
* If your GPU is not a Volta, edit `CCFLAGS` and `KOKKOS_ARCH` options in `src/MAKE/OPTIONS/Makefile.kokkos_cuda_mpi` and `Dockerfile` based on the following table: 

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

## Run the Application
Note that to successfully build this docker image and the necessary libraries/packages used for PageRank, you will
need sudo access on the machine you are doing this work. Otherwise, the container image will fail to build.

There are 5 steps to take to run ResNet-50 successfully. 
1. Download the ImageNet data set from [this link](https://image-net.org/download-images). We do not provide the data set in this artifact repo because it is so large. 
2. Update lines 28 and 29 in `run-resnet.sh` to provide the directory where the training data set and validation data set is located on your machine.
3. Run `sudo docker build -t resnet_image .`
4. Run `sudo docker run --gpus all resnet_image`
5. Move data output by profiler (nvprof) from container to local directory in this repository - see below. 

There will be 4 csv files and 1 txt file output by the profiler (nvprof), which contains kernel information, GPU SM frequency, power, and temperature. These files will be stored in the docker container by default. To access these files, you will have to copy them using `docker cp` (step 5) to the directory of your choice (we recommend `../out/`).

```
sudo docker create -ti --name dummy resnet_image bash
<Returns Container ID c_id>
sudo docker cp <c_id>:/sec5a/*.csv ../out/.
sudo docker rm -f dummy``
```

## Build and Run Without Docker
There are 4 steps to take to run ResNet-50 using shell scripts:
1. Download the ImageNet data set from [this link](https://image-net.org/download-images). We do not provide the data set in this artifact repo because it is so large. 
2. Update lines 28 and 29 in `run-resnet.sh` to provide the directory where the training data set and validation data set is located on your machine.
3. Run `chmod u+x run-resnet.sh`.
4. Run `run-resnet.sh`.

You will find a few output files in `../out`:
  - `out/resnet_*.csv`: contains kernel information, GPU SM frequency, power, and temperature. There will be one csv file per GPU (e.g., if you trained on 4 GPUs, there will be 4 csv files).
  - `out/resnet_iterdur_*.txt`: contains iteration durations. Iteration durations are directly printed from line 75 in `sec5a_resnet/cnn_utils/engine.py`. Only one text file. 
