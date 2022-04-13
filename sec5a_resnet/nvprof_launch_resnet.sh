#!/bin/bash

NGPUS=4
NNODES=1
MASTER="c006-009"
MVAPICH=false

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        --help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  --help           Display this help message"
            echo "  --ngpus [count]  Number of GPUs per node (default: 1)"
            echo "  --nnodes [count] Number of nodes this script is launched on (default: 1)"
            echo "  --master [addr]  Address of master node (default: \"\")"
            echo "  --mvapich           Use MVAPICH env variables for initialization (default: false)"
            exit 0
        ;;
        --ngpus)
            NGPUS=$VALUE
        ;;
        --nnodes)
            NNODES=$VALUE
        ;;
        --master)
            MASTER=$VALUE
        ;;
        --mvapich)
            MVAPICH=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

if [ "$MVAPICH" == true ]; then
  LOCAL_RANK=$MV2_COMM_WORLD_RANK
else
  LOCAL_RANK=$OMPI_COMM_WORLD_RANK
fi

#module load conda
#conda deactivate
#conda activate pytorch
#module unload spectrum_mpi
#module use /home/01255/siliu/mvapich2-gdr/modulefiles/
#module load gcc/7.3.0 
#module load mvapich2-gdr/2.3.4

#export MV2_USE_CUDA=1
#export MV2_ENABLE_AFFINITY=1
#export MV2_THREADS_PER_PROCESS=2
#export MV2_SHOW_CPU_BINDING=1
#export MV2_CPU_BINDING_POLICY=hybrid
#export MV2_HYBRID_BINDING_POLICY=spread
#export MV2_USE_RDMA_CM=0
#export MV2_SUPPORT_DL=1

export OMP_NUM_THREADS=4

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH
echo ---------------------------------------------------------------------------------------------------------------------------------------
KWARGS=""
KWARGS+="--base-lr 0.05 "
KWARGS+="--batch-size 64 "
# Vary batch size to make workload "bigger"
# What is 128? Per GPU batch size or across all GPUs? 128 is too high on one GPU (this is probably 4 GPU number we think)
KWARGS+="--kfac-update-freq 0 "
#KWARGS+="--kfac-update-freq 500 "
KWARGS+="--kfac-cov-update-freq 50 "
KWARGS+="--damping 0.003 "
KWARGS+="--model resnet50 "
KWARGS+="--checkpoint-freq 90 "
KWARGS+="--kfac-comm-method comm-opt "
KWARGS+="--kfac-grad-worker-fraction 0.0 "
KWARGS+="--train-dir /tmp/imagenet/ILSVRC2012_img_train "
KWARGS+="--val-dir /tmp/imagenet/ILSVRC2012_img_val "
KWARGS+="--log-dir logs "
#KWARGS+="--fp16 "

# KFAC Schedule
#KWARGS+="--epochs 55 "
#KWARGS+="--lr-decay 25 35 40 45 50 "
# SGD Schedule
KWARGS+="--epochs 1 "
KWARGS+="--lr-decay 30 60 80 "

# Check if node directory data exists, if not create it
if [ -d "/scratch/08022/psinha9/kfac_pytorch/nvprof-outputs/${HOSTNAME}-data" ]
then
    echo "Node data directory exists"
else
    mkdir /scratch/08022/psinha9/kfac_pytorch/nvprof-outputs/${HOSTNAME}-data
fi

# Get timestamp and device number (0, 1, 2, 3) for the node that is running ResNet
ts=`date '+%s'`

if [ $NNODES -eq 1 ]; then
  __PREFETCH=off /usr/local/cuda-10.1/bin/nvprof --print-gpu-trace \
      --profile-child-processes \
      --system-profiling on --kernel-latency-timestamps on \
      --csv --log-file /scratch/08022/psinha9/kfac_pytorch/nvprof-outputs/${HOSTNAME}-data/resnet_%p_${ts}_${HOSTNAME}.csv \
      --device-buffer-size 128 \
      --continuous-sampling-interval 1 \
      -f python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
    /scratch/08022/psinha9/kfac_pytorch/examples/torch_imagenet_resnet.py $KWARGS > /scratch/08022/psinha9/kfac_pytorch/nvprof-outputs/${HOSTNAME}-data/resnet_iterdur_${ts}_${HOSTNAME}.txt
else
  __PREFETCH=off /usr/local/cuda-10.2/bin/nvprof --print-gpu-trace \
      --profile-child-processes \
      --system-profiling on --kernel-latency-timestamps on \
      --csv --log-file resnet_%p_${ts}_${HOSTNAME}.csv \
      --device-buffer-size 128 \
      --continuous-sampling-interval 1 \
      -f python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
      --nnodes=$NNODES \
      --node_rank=$LOCAL_RANK \
      --master_addr=$MASTER \
    examples/torch_imagenet_resnet.py $KWARGS
fi                                                                                                                                                                                                              

echo Application complete!
echo ----------------------------------------------------------------------------------
