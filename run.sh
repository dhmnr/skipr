#!/bin/bash
export TASK_NAME=qnli
export EPOCHS=25

poetry run glue \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --output_dir ./models/$TASK_NAME/classifier \
  --overwrite_output_dir \
  --mode classifier

# for i in $(seq 0.05 0.01 0.1); do
# i=0.05
    poetry run glue \
      --model_name_or_path ./models/$TASK_NAME/classifier \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-4 \
      --num_train_epochs $EPOCHS \
      --output_dir ./models/$TASK_NAME/policy \
      --overwrite_output_dir \
      --mode policy \
      --alpha 0.07
# done

