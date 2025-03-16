import numpy as np

# 加载 .npy 文件，并允许加载对象类型的数据
metrics = np.load('results/Electricity_48_1240_Multiscale_DRPK_custom_ftM_sl48_pl1240_cycle168_cycle_pattern_daily+weekly+monthly+yearly_nums_4_mlp_seed2024/metrics.npy', allow_pickle=True)

# 查看数据内容
print(metrics)

print(f"数据类型: {metrics.dtype}")
