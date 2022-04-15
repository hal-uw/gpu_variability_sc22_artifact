# Section 5A: ResNet-50 on NVIDIA GPUs

## Application Overview and Directory Structure

We ran the training phase of ResNet-50 CNN. We chose the 50-layer version because it is a stable, commonly used benchmark in the HPC community. Our training set was 1.2 million images from ImageNet and our batch size was 64. We define one training run as 500 iterations. Note that we did not complete training on the entire training set; 500 training iterations was sufficiently long to collect profiling data while training was stable. Our training ran across four GPUs on one node. Because we only use one node, we do not need to use any `mpi` commands. Below is a breakdown of the this directory. 
```
├── sec5a_resnet
│   ├── cnn_utils: a directory containing utility files used in the pytorch resnet implementation
│   ├── torch_imagenet_resnet.py: main python file called to launch ResNet
│   ├── utils.py: utility functions imported into torch_imagenet_resnet.py
│   ├── README.md: contains ResNet-50 specific instructions on running the application and adjusting input configurations
│   ├── run-resnet.sh: runs ResNet-50 on NVIDIA GPUs (UPDATE BEFORE RUNNING)
```

## Adjusting Input Configurations
To adjust configuration parameters, update `run-resnet.sh`. Specifically, update `NGPUS` and `NNODES` to adjust the number of gpus and/or nodes, respectively. Update line 19 to adjust the batch size. The default number of gpus is 4, number of nodes is 1, and batch size is 64. Finally, to adjust the number of training iterations, change line 38 in `sec5a_resnet/cnn_utils/engine.py`.

## Running ResNet-50 on NVIDIA GPUs
There are 3 steps to take to run ResNet-50 successfully. 
1. Download the ImageNet data set from [this link](https://image-net.org/download-images). We do not provide the data set in this artifact repo because it is so large. 
2. Update line 28 to provide the directory where the training data set is located on your machine.
3. Run `run-resnet.sh`.

You will find two output files in `../out`:
  - `out/resnet_*.csv`: contains kernel information, GPU SM frequency, power, and temperature
  - `out/resnet_iterdur_*.txt`: contains iteration durations. Iteration durations are directly printed from line 75 in `sec5a_resnet/cnn_utils/engine.py`.

