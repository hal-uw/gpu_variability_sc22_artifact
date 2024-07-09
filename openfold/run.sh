python /opt/openfold/train_openfold.py \
        /scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
        /scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/alignment_openfold \
        /scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
        full_output \
        2021-10-10 \
        --val_data_dir /scratch1/00946/zzhang/datasets/openfold/openfold/cameo/mmcif_files \
        --val_alignment_dir /scratch1/00946/zzhang/datasets/openfold/openfold/cameo/alignments \
        --template_release_dates_cache_path=/scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/mmcif_cache.json \
        --precision=32 \
        --train_epoch_len 128000 \
        --gpus=4 \
        --num_nodes=1 \
        --accumulate_grad_batches 8 \
        --replace_sampler_ddp=True \
        --seed=7152022 \
        --deepspeed_config_path=/scratch1/00946/zzhang/frontera/openfold/deepspeed_config.json \
        --checkpoint_every_epoch \
        --obsolete_pdbs_file_path=/scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/pdb_mmcif/obsolete.dat \
        --train_chain_data_cache_path=/scratch1/00946/zzhang/datasets/openfold/openfold/ls6-tacc/chain_data_cache.json
