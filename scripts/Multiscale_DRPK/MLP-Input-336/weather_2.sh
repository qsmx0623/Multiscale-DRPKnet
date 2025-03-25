model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Weather/
data_path_name=Weather.csv
model_id_name=weather
data_name=custom
cycle_pattern=daily+weekly
pattern_nums=2

model_type='mlp'
seq_len=336
for pred_len in 96 192 336 720 960 1024 1240 1688
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --steps_per_day 144 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --cycle_pattern $cycle_pattern \
      --pattern_nums $pattern_nums \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 10 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed \
      --gpu 5 \
      --device '5,6,7' \
      --use_multi_gpu
done
done

