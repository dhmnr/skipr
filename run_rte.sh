#!/bin/bash
export TASK_NAME=rte

poetry run glue \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --mode classifier