#!/bin/bash

# weights and bias API key
export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/opt/python3.10/site-packages
TORCHPATH=/opt/python3.10/site-packages/bin

export MASTER_PORT=$(shuf -i 30000-60000 -n1)

OUTPUT_DIR=${HOME}/GPT_DDP_weights

${TORCHPATH}/torchrun --nproc_per_node 2 --master_port=${MASTER_PORT} ../run_clm.py \
  --overwrite_output_dir \
  --fp16 True \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ${OUTPUT_DIR} \
  --per_device_train_batch_size 1 \
  --max_steps 200 \
  --do_train
