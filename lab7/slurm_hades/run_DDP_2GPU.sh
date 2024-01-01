#!/bin/bash
#SBATCH --job-name=gpt2_multi      ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --ntasks-per-node=1      ## 每個節點運行 1 srun tasks
#SBATCH --cpus-per-task=12       ## 每個 srun task 索取 16 CPUs
#SBATCH --gres=gpu:2            ## 每個節點索取 2 GPUs
#SBATCH -o %j.out           # Path to the standard output file
#SBATCH -e %j.err           # Path to the standard error ouput file

# weights and bias API key
export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/opt/python3.10/site-packages
TORCHPATH=/opt/python3.10/site-packages/bin

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 30000-60000 -n1)

echo "NODELIST="${SLURM_NODELIST}
echo "MASTER_ADDR="${MASTER_ADDR}
echo "MASTER_PORT="${MASTER_PORT}

OUTPUT_DIR=${HOME}/GPT_DDP_weights

srun ${TORCHPATH}/torchrun --nnodes 1 --nproc_per_node 2 \
  --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} \
  ../run_clm.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --learning_rate=5e-5 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ${OUTPUT_DIR} \
  --per_device_train_batch_size 1 \
  --max_steps 200 \
  --do_train
