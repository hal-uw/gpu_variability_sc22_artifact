#!/bin/bash

NGPUS=1
NNODES=1
LOCAL_RANK=""
MASTER=""
KWARGS=""
RUN=-1

while [[ "$1" == -* ]]; do
    case "$1" in
        -h|--help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  -h,--help           Display this help message"
            echo "  -N,--ngpus  [int]   Number of GPUs per node (default: 1)"
            echo "  -n,--nnodes [int]   Number of nodes this script is launched on (default: 1)"
            echo "  -r,--rank   [int]   Node rank (default: \"\")"
            echo "  -m,--master [str]   Address of master node (default: \"\")"
            echo "  -a,--kwargs [str]   Training arguments. MUST BE LAST ARG! (default: \"\")"
            exit 0
        ;;
        -N|--ngpus)
            shift
            NGPUS="$1"
        ;;
        -n|--nnodes)
            shift
            NNODES="$1"
        ;;
        -m|--master)
            shift
            MASTER="$1"
        ;;
        -r|--rank)
            shift
            LOCAL_RANK="$1"
        ;;
        -a|--kwargs)
            shift
            KWARGS="$@"
        ;;
        -r|--run)
            shift
            RUN="$1"
        ;;
        *)
          echo "ERROR: unknown parameter \"$1\""
          exit 1
        ;;
    esac
    shift
done

#source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh
#conda activate bert-pytorch

if [[ -z "$LOCAL_RANK" ]]; then
    if [[ -z "${OMPI_COMM_WORLD_RANK}" ]]; then
        LOCAL_RANK=${MV2_COMM_WORLD_RANK}
    else
        LOCAL_RANK=${OMPI_COMM_WORLD_RANK}
    fi
fi

NUM_THREADS=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
#export OMP_NUM_THREADS=$((NUM_THREADS / NGPUS))

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, host=$HOSTNAME

ts=`date '+%s'`
if [[ "$NNODES" -ne 1 ]]; then
    echo "Current setup only handles running BERT-pretraining on a single node. Aborting."
    exit
fi

#if [ -d "/usr/local/cuda-10.1/bin" ]
#then
#    __PREFETCH=off /usr/local/cuda-10.1/bin/nvprof --print-gpu-trace \
#        --openacc-profiling off \
#        --profile-child-processes \
#        --system-profiling on \
#        --kernel-latency-timestamps on \
#        --device-buffer-size 128 \
#        --continuous-sampling-interval 1 \
#        --csv --log-file bert_%p_${ts}_${HOSTNAME}.csv \
#        --device-buffer-size 128 -f \
#        python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#        run_pretraining.py $KWARGS
#elif [ -d "/usr/local/cuda/bin" ]
#then
#    echo "WARNING: Could not find cuda-10.1, but found a cuda installation. Power measurements may not be included."
#    __PREFETCH=off /usr/local/cuda/bin/nvprof --print-gpu-trace \
#        --openacc-profiling off \
#        --profile-child-processes \
#        --system-profiling on \
#        --kernel-latency-timestamps on \
#        --device-buffer-size 128 \
#        --continuous-sampling-interval 1 \
#        --csv --log-file bert_%p_${ts}_${HOSTNAME}.csv \
#        --device-buffer-size 128 -f \
#        python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#        run_pretraining.py $KWARGS
#else 

#    echo "Couldn't find CUDA bin directory. Check /usr/local for your CUDA installation and update run-sgemm-nvidia.sh. Aborting."
#fi

touch data/bert_multi_iterdur_${ts}_${HOSTNAME}_run_$RUN.txt

__PREFETCH=off python -m torch.distributed.launch \
 --nproc_per_node=$NGPUS \
run_pretraining.py $KWARGS > data/bert_multi_iterdur_${ts}_${HOSTNAME}_run_$RUN.txt

