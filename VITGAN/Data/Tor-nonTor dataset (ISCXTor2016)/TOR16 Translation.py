import pandas as pd

# 读取两个CSV文件
scenario_a = pd.read_csv('Scenario-A-merged_5s.csv')
scenario_b = pd.read_csv('Scenario-B-merged_5s.csv')

# 合并这两个文件
merged_data = pd.concat([scenario_a, scenario_b], ignore_index=True)

# 保存为新的CSV文件
merged_data.to_csv('data1.csv', index=False)