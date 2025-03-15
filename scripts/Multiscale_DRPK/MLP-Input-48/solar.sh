model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Solar-Energy/
data_path_name=Solar-Energy.csv
model_id_name=SolarEnergy
data_name=custom
cycle_pattern=hourly+daily+weekly+monthly+yearly
pattern_nums=5
gpu_id=6

model_type='mlp'
seq_len=48
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
      --gpu $gpu_id \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --cycle 144 \
      --cycle_pattern $cycle_pattern\
      --pattern_nums $pattern_nums\
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.005 --random_seed $random_seed
done
done


