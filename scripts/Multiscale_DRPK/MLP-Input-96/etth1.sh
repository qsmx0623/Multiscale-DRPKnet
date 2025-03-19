model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/ETTh1/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
cycle_pattern=daily+weekly

model_type='mlp'
seq_len=96
for pred_len in 96
do
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
      --enc_in 7 \
      --cycle 24 \
      --cycle_pattern $cycle_pattern\
      --model_type $model_type \
      --train_epochs 30 \
      --patience 10 \
      --gpu 5\
      --itr 1 --batch_size 256 --learning_rate 0.005
done