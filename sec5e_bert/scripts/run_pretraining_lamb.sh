#!/bin/bash

# Change to 2 for Phase 2 training
PHASE=1

if [[ "$PHASE" -eq 1 ]]; then
        CONFIG=config/bert_pretraining_phase1_config.json
        DATA=/scratch/00946/zzhang/data/bert/bert_masked_wikicorpus_en/phase1
else
        CONFIG=config/bert_kfac_pretraining_phase2_config.json
        DATA=/scratch/00946/zzhang/data/bert/bert_masked_wikicorpus_en/phase2
fi

mkdir -p sbatch_logs

HOSTFILE=hostfile
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=4

# PHASE 1
#mpiexec -N 1 -hostfile $HOSTFILE \
bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --kwargs \
    --input_dir $DATA \
    --output_dir results/bert_pretraining \
    --config_file $CONFIG \
    --num_steps_per_checkpoint 200 \
    --global_batch_size 16 \
    --local_batch_size 4 

# PHASE 2
#mpirun -np $NODES -hostfile $HOSTFILE  bash scripts/launch_pretraining.sh  \
#    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
#    --config config/bert_pretraining_phase2_config.json \
#	--input data/hdf5/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/ \
#    --output results/bert_pretraining

