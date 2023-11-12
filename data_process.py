import pandas as pd

# 读取CSV文件
df = pd.read_csv('datasets/training_dataset/training_3_category_1/training_3_category_1.csv')

# 按照"aircraft"列和"timeAtServer"列排序
df = df.sort_values(by=['aircraft', 'timeAtServer'])

# 通过"aircraft"列进行分组
grouped = df.groupby('aircraft')

# 如果您想要将每个分组的数据存储在一个字典中，可以使用以下代码：
grouped_data = {aircraft: group for aircraft, group in grouped}

# # 如果您想要将每个分组的数据存储在一个新的CSV文件中，可以使用以下代码：
# for aircraft, group in grouped:
#     group.to_csv(f'{aircraft}_sorted.csv', index=False)

# 如果您需要将所有分组的数据合并为一个DataFrame，可以使用以下代码：
sorted_df = pd.concat([group for _, group in grouped], ignore_index=True)

# 保存排序后的数据到新的CSV文件
sorted_df.to_csv('sorted_3.csv', index=False)
