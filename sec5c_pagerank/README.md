# Section 5D: PageRank on NVIDIA GPUs



### Pre-Requisites
* Machine with an NVIDIA GPU
* Relevant GPU drivers installed (nvidia-smi)
* Compilation and launch scripts assume a Volta V100 Class GPU (arch70). If your GPU is not a Volta, edit `src/cuda/pannotia/pagerank_mod/Makefile` and uncomment the relevant `GENCODE` line based on the table below:
| NVIDIA Architecture        | Cards                                 | Supported Sm and Gencode Variations |
|:---------------------------|:--------------------------------------|:------------------------------------|
| Fermi (CUDA 3.2 until 8)   | GeForce 400, 500, 600, GT-630         | `SM_20` `compute_30`                |
| Kepler(CUDA 5 until 10)    | GeForce 700, GT-730                   | `SM_30` `compute_30`                |
|                            | Tesla K40                             | `SM_35` `compute_35`                |
|                            | Tesla K80                             | `SM_37` `compute_37`                |
| Maxwell (CUDA 6 until 11)  | Tesla/Quadro M                        | `SM_50` `compute_50`                |
|                            | Quadro M6000, GeForce 900,            | `SM_52` `compute_52`                |
|                            | GTX-970, GTX-980, GTX Titan-X         |                                     |
|                            | Jetson TX1/Tegra X1,                  | `SM_53` `compute_53`                |
|                            | Drive CX/PX, Jetson Nano              |                                     |
| Pascal (>= CUDA 8)         | Quadro GP100, Tesla P100, DGX-1       | `SM_60` `compute_60`                |
|                            | GTX 1080/1070/1060/1050/1030/1010     | `SM_61` `compute_61`                |
|                            | Titan Xp, Tesla P40, Tesla P4         |                                     |
|                            | iGPU on Drive PX2, Tegra(Jetson) TX2  | `SM_62` `compute_62`                |
| Volta (>= CUDA 9)          | **Tesla V100**, DGX-1, GTX1180,       | `SM_70` `compute_70`                |
|                            | Titan V, Quadro GV100                 |                                     |
|                            | Jetson AGX Xavier, Drive AGX Pegasus  | `SM_72` `compute_72`                |
|                            | Xavier NX                             |                                     |
| Turing (>= CUDA 10)        | GTX 1660, RTX 20X0 (X=6/7/8), Titan RTX, | `SM_75` `compute_75`                |
|                            | Quadro RTX 4000/5000/6000/8000/T1000/T2000,|                                 |
|                            | Tesla T4                              |                                     |
| Ampere (>= CUDA 11.1)      | A100, GA100, DGX-A100                 | `SM_80` `compute_80`                |
|                            | GA10X cards, RTX 30X0 (X=5/6/7/8/9)   | `SM_86` `compute_86`                |

### Build Container Image
```
# Build container image
docker build -t pagerank_img .
# Run application
docker run --gpus all pagerank_img
# Prints the profiling log name <csv_name>, move to local storage
docker cp pagerank_img:<csv name> .
```

### Run the application
