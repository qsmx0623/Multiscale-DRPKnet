import pandas as pd

# 读取使用制表符分隔的txt文件
txt_file_path = 'test_results/weather_336_96_TERNet_custom_ftM_sl336_pl96_pattern144_cycle_pattern_daily+weekly+monthly+yearly_nums_4_mlp_seed2024/80true.txt'
df = pd.read_csv(txt_file_path, delimiter='\t')  # 使用制表符 '\t' 作为分隔符

# 将数据保存为Excel文件
excel_file_path = 'test_results/weather_336_96_TERNet_custom_ftM_sl336_pl96_pattern144_cycle_pattern_daily+weekly+monthly+yearly_nums_4_mlp_seed2024/80_true.xlsx'
df.to_excel(excel_file_path, index=False, engine='openpyxl')

print(f"文件已成功保存为 {excel_file_path}")
