#!/bin/bash

model_name=RLinear
root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/ExchangeRate/
data_path_name=ExchangeRate.csv
model_id_name=ExchangeRate
data_name=custom
seq_len=336

# 定义要循环的pred_len值
pred_lens=(96 192 336 720)

# 遍历pred_len值
for pred_len in "${pred_lens[@]}"
do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id_name $model_id_name \
        --model_id "${model_id_name}_${seq_len}_${pred_len}" \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 8 \
        --train_epochs 5 \
        --itr 1 \
        --batch_size 64 \
        --random_seed 2024 \
        --gpu 2 \
        --device '2,6' \
        --use_multi_gpu
done
