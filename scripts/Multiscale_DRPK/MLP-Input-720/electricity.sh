model_name=Multiscale_DRPK

root_path_name=/home/home_new/qsmx/pycodes/BasicTS/datasets/raw_data/Electricity/
data_path_name=Electricity.csv
model_id_name=Electricity
data_name=custom
cycle_pattern=daily+weekly+monthly+yearly
gpu_id=7

model_type='mlp'
seq_len=720
for pred_len in 1688
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
      --gpu $gpu_id \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --cycle 168 \
      --cycle_pattern $cycle_pattern\
      --pattern_nums 4\
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.005 --random_seed $random_seed
      
done
done


