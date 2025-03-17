model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Traffic/
data_path_name=Traffic.csv
model_id_name=Traffic
data_name=custom
cycle_pattern=daily+weekly+monthly+yearly

model_type='mlp'
seq_len=96
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
      --freq h\
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --cycle_pattern $cycle_pattern\
      --pattern_nums 4\
      --model_type $model_type \
      --train_epochs 1 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.005 --random_seed $random_seed \
      --use_gpu True \
      --gpu 5\
      --devices '5,6' \
      --use_multi_gpu 
done
done