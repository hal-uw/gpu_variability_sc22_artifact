#!/bin/bash

# We only run Phase 1 of BERT pretraining

CONFIG=config/bert_pretraining_phase1_config.json
DATA=data/encoded/sequences_lowercase_max_seq_len_512_next_seq_task_true

mkdir -p sbatch_logs

MASTER_RANK=0
NODES=1
PROC_PER_NODE=4

# PHASE 1
bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --kwargs \
    --input_dir $DATA \
    --output_dir results/bert_pretraining \
    --config_file $CONFIG \
    --num_steps_per_checkpoint 200 \
    --global_batch_size 16 \
    --local_batch_size 4 