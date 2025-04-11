model_name=TERNet

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/ETTh1/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
cycle_pattern=daily
pattern_nums=1

model_type='mlp'
seq_len=96
for pred_len in 336 720 960 1024 1240 1688
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model_id_name $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --steps_per_day 24 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --pattern 24 \
      --cycle_pattern $cycle_pattern \
      --pattern_nums $pattern_nums \
      --model_type $model_type \
      --train_epochs 50 \
      --patience 10 \
      --itr 1 --batch_size 64 --random_seed $random_seed \
      --gpu 0 \
      --device '0,2' \
      --use_multi_gpu 
done
done

