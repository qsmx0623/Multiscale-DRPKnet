#!/bin/bash

model_name=TERNet_DLinear
root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/ExchangeRate/
data_path_name=ExchangeRate.csv
model_id_name=ExchangeRate
data_name=custom
seq_len=336

# 定义要循环的cycle_pattern和pattern_nums
cycle_patterns=("daily+weekly")
pattern_nums=(2)

# 定义要循环的pred_len值
pred_lens=(96 192 336 720)

# 结果文件
results_file="result.txt"

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
            # 构建搜索模式
            search_pattern="${model_id_name}_${seq_len}_${pred_len}_${model_name}_${data_name}_ftM_sl${seq_len}_pl${pred_len}_cycle12_cycle_pattern_${cycle_pattern}_nums_${pattern_num}_seed${random_seed}"
            
            # 检查结果文件是否包含该模式
            if grep -q "$search_pattern" "$results_file"; then
                echo "Results already exist for $search_pattern, skipping..."
            else
                # 运行python脚本
                python -u run.py \
                  --is_training 1 \
                  --root_path $root_path_name \
                  --data_path $data_path_name \
                  --model_id_name $model_id_name \
                  --model_id $model_id_name'_'$seq_len'_'$pred_len \
                  --model $model_name \
                  --data $data_name \
                  --features M \
                  --steps_per_day 1 \
                  --seq_len $seq_len \
                  --pred_len $pred_len \
                  --enc_in 8 \
                  --pattern 12 \
                  --cycle_pattern $cycle_pattern \
                  --pattern_nums $pattern_num \
                  --train_epochs 30 \
                  --patience 10 \
                  --itr 1 --batch_size 64 --random_seed $random_seed \
                  --gpu 2 \
                  --device '2,3,4' \
                  --use_multi_gpu 
            fi
        done
    done
done
