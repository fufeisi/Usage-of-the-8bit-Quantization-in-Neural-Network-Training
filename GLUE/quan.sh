#!/bin/bash

#SBATCH --time 336:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

export QUAN=1
echo 'Quantize the activation maps!'
for TASK_NAME in cola sst2 mrpc stsb qqp mnli qnli rte wnli
do
     echo $TASK_NAME
     python main.py --quan $QUAN --model_name_or_path roberta-large --task_name $TASK_NAME --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 1 --output_dir train/$TASK_NAME/$QUAN
done
