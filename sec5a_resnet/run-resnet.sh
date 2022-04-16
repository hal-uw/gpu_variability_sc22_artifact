#!/bin/bash

NGPUS=4
NNODES=1
MVAPICH=false

if [ "$MVAPICH" == true ]; then
  LOCAL_RANK=$MV2_COMM_WORLD_RANK
else
  LOCAL_RANK=$OMPI_COMM_WORLD_RANK
fi

export OMP_NUM_THREADS=4

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH
echo ---------------------------------------------------------------------------------------------------------------------------------------
KWARGS=""
KWARGS+="--base-lr 0.05 "
KWARGS+="--batch-size 64 "
KWARGS+="--kfac-update-freq 0 "
#KWARGS+="--kfac-update-freq 500 "
KWARGS+="--kfac-cov-update-freq 50 "
KWARGS+="--damping 0.003 "
KWARGS+="--model resnet50 "
KWARGS+="--checkpoint-freq 90 "
KWARGS+="--kfac-comm-method comm-opt "
KWARGS+="--kfac-grad-worker-fraction 0.0 "
KWARGS+="--train-dir [provide path to training image set directory here (e.g, /tmp/imagenet/imagenet-tiny/ILSVRC2012_img_train)] "
KWARGS+="--val-dir [providing validation directory (e.g., /tmp/imagenet/imagenet-tiny/ILSVRC2012_img_val)] "
# KWARGS+="--log-dir logs "
#KWARGS+="--fp16 "

# KFAC Schedule
#KWARGS+="--epochs 55 "
#KWARGS+="--lr-decay 25 35 40 45 50 "
# SGD Schedule
KWARGS+="--epochs 1 "
KWARGS+="--lr-decay 30 60 80 "

# Get timestamp and device number (0, 1, 2, 3) for the node that is running ResNet
ts=`date '+%s'`

__PREFETCH=off /usr/local/cuda-10.1/bin/nvprof --print-gpu-trace \
    --profile-child-processes \
    --system-profiling on --kernel-latency-timestamps on \
    --csv --log-file resnet_%p_${ts}_${HOSTNAME}.csv \
    --device-buffer-size 128 \
    --continuous-sampling-interval 1 \
    -f python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
torch_imagenet_resnet.py $KWARGS > resnet_iterdur_${ts}_${HOSTNAME}.txt                                                                                                                                                                                                        
echo resnet_%p_${ts}_${HOSTNAME}.csv
echo resnet_iterdur_${ts}_${HOSTNAME}.txt
echo Completed ResNet Run
echo ResNet-50 Application complete!
echo ----------------------------------------------------------------------------------
