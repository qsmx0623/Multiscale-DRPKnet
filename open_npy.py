import numpy as np

# 加载 .npy 文件，并允许加载对象类型的数据
x = np.load('results/weather_336_96_TERNet_custom_ftM_sl336_pl96_pattern144_cycle_pattern_daily+weekly+monthly+yearly_nums_4_mlp_seed2024/true.npy', allow_pickle=True)

# 查看数据内容
print(x)
print(x.shape)
print(f"数据类型: {x.dtype}")
