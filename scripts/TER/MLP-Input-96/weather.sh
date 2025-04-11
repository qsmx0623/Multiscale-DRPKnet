model_name=TERNet

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Weather/
data_path_name=Weather.csv
model_id_name=weather
data_name=custom
cycle_pattern=daily+monthly+yearly
pattern_nums=3

model_type='mlp'
seq_len=96
for pred_len in 960 1024 1240 1688
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
      --steps_per_day 144 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --pattern 144 \
      --cycle_pattern $cycle_pattern \
      --pattern_nums $pattern_nums \
      --model_type $model_type \
      --train_epochs 50 \
      --patience 10 \
      --itr 1 --batch_size 64 --random_seed $random_seed \
      --gpu 2 \
      --device '2,3,4,5,6' \
      --use_multi_gpu 
done
done

