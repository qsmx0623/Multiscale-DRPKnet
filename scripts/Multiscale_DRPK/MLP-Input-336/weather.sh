model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Weather/
data_path_name=Weather.csv
model_id_name=weather
data_name=custom
cycle_pattern=hourly+daily+weekly+monthly+yearly

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
      --freq 10min\
      --features M \
      --gpu 1 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --cycle_pattern $cycle_pattern\
      --pattern_nums 5\
      --model_type $model_type \
      --train_epochs 50 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
done
done

