#!/bin/bash

SCRIP_DIR=$(echo `cd $(dirname $0); pwd`)
export PATH=/work/cache/env/miniconda3/bin:$PATH

export TRAIN_DATASET=outputs/v1_20240713/emb_samples_qd_v2.jsonl

export N_EPOCH=2
export TRAIN_GROUP_SIZE=32

export GRADIENT_ACCUMULATION_STEPS=32
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export N_NODES=1
export BATCH_SIZE=`expr ${GRADIENT_ACCUMULATION_STEPS} \* ${PER_DEVICE_TRAIN_BATCH_SIZE} \* ${N_NODES}`

export VERSION=ft_v1_bge_base_epoch_${N_EPOCH}_bz_${BATCH_SIZE}_trgrp_${TRAIN_GROUP_SIZE}_$(date +"%Y%m%d_%H%M")

# 如果需要将训练进度上传Wandb，可以取消下方注释
#export WANDB_PROJECT=RAG-From-Scratch-Reranker-Finetune
#export WANDB_API_KEY=替换为自己的Key
#export WANDB_NAME=${VERSION}

export OUTPUT_DIR=experiments/reranker/finetune/${VERSION}

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

torchrun --nproc_per_node ${N_NODES} \
-m FlagEmbedding.reranker.run \
--output_dir ${OUTPUT_DIR} \
--model_name_or_path BAAI/bge-reranker-base \
--train_data ${TRAIN_DATASET} \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs ${N_EPOCH} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--dataloader_drop_last True \
--train_group_size ${TRAIN_GROUP_SIZE} \
--logging_steps 5 \
--save_steps 50 \
--save_total_limit 10 \
--warmup_ratio 0.05 \
--lr_scheduler_type cosine


cp "$SCRIP_DIR/$0" ${OUTPUT_DIR}
