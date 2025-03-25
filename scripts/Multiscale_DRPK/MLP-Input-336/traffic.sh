#!/bin/bash

model_name=Multiscale_DRPK
root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Traffic/
data_path_name=Traffic.csv
model_id_name=traffic
data_name=custom
seq_len=336
model_type='mlp'

# 定义要循环的cycle_pattern和pattern_nums
cycle_patterns=("daily" "daily+weekly" "daily+weekly+monthly" "daily+weekly+monthly+yearly")
pattern_nums=(1 2 3 4)

# 定义要循环的pred_len值
pred_lens=(96 192 336 720 960 1024 1240 1688)

# 遍历cycle_patterns和pattern_nums
for i in "${!cycle_patterns[@]}"
do
    cycle_pattern=${cycle_patterns[$i]}
    pattern_num=${pattern_nums[$i]}

    # 遍历pred_len值
    for pred_len in "${pred_lens[@]}"
    do
        # 遍历random_seed值
        for random_seed in 2024
        do
            # 运行python脚本
            python -u run.py \
              --is_training 1 \
              --root_path $root_path_name \
              --data_path $data_path_name \
              --model_id $model_id_name'_'$seq_len'_'$pred_len \
              --model $model_name \
              --data $data_name \
              --features M \
              --steps_per_day 24 \
              --seq_len $seq_len \
              --pred_len $pred_len \
              --enc_in 862 \
              --cycle 168 \
              --cycle_pattern $cycle_pattern \
              --pattern_nums $pattern_num \
              --model_type $model_type \
              --train_epochs 30 \
              --patience 10 \
              --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
              --gpu 2 \
              --device '2,3,5' \
              --use_multi_gpu
        done
    done
done
