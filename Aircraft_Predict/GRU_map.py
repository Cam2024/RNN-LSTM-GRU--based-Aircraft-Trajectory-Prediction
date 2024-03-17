import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 数据读取与预处理
training_set = pd.read_csv('dataset.csv')  # 替换成你的数据文件路径
training_set = training_set.iloc[:, 1:].values

# 数据归一化
sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

# 定义滑动窗口函数
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

seq_length = 30  # 序列长度，根据你的需求设定

# 应用滑动窗口处理数据
x, y = sliding_windows(training_data, seq_length)

# 划分训练集和测试集
train_size = int(len(y) * 0.9)

dataX = torch.Tensor(x).cuda()
dataY = torch.Tensor(y).cuda()

trainX = dataX[:train_size]
trainY = dataY[:train_size]

testX = dataX[train_size:]
testY = dataY[train_size:]

# GRU模型定义
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 设置训练参数
num_epochs = 300
learning_rate = 0.01

input_size = 3  # 输入维度为3，即latitude, longitude, geoAltitude
hidden_size = 128  # 根据需求设定
num_layers = 2  # 减少隐藏层数量
output_size = 3  # 输出维度为3，即latitude, longitude, geoAltitude

# 实例化模型
gru = GRU(input_size, hidden_size, num_layers, output_size).cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(gru.parameters(), lr=learning_rate)

# 训练模型
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    gru.train()
    outputs = gru(trainX)
    optimizer.zero_grad()
    train_loss = criterion(outputs, trainY)
    train_losses.append(train_loss.item())
    train_loss.backward()
    optimizer.step()
    # 验证集上的损失
    with torch.no_grad():
        gru.eval()
        test_outputs = gru(testX)
        test_loss = criterion(test_outputs, testY)
        test_losses.append(test_loss.item())
    print("Epoch: %d, loss: %1.5f" % (epoch, math.sqrt(train_loss.item())))

# 计算RMSE并绘制图表
train_rmse = torch.sqrt(torch.tensor(train_losses))  # 计算训练集RMSE
test_rmse = torch.sqrt(torch.tensor(test_losses))    # 计算测试集RMSE

# 绘制损失图表
plt.plot(train_rmse, label='Train Loss')
plt.plot(test_rmse, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Root Mean Squared Error (RMSE)')
plt.legend()
plt.show()

# 模型评估
gru.eval()
train_predict = gru(dataX).cpu().data.numpy()
data_predict = sc.inverse_transform(train_predict)
dataY_plot = sc.inverse_transform(dataY.cpu().data.numpy())

# 可视化结果
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置地图的范围为整个欧洲
ax.set_extent([-12, 40, 35, 70], crs=ccrs.PlateCarree())

# 添加地图特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgrey')

# 绘制经纬度轨迹
scatter = ax.scatter(training_set[:, 1], training_set[:, 0], c='blue', cmap='viridis', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='True Trajectory', s = 10)

# 绘制预测轨迹
ax.scatter(data_predict[:train_size, 1], data_predict[:train_size, 0], c='green', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='Predicted Trajectory (Train)', s = 10)
ax.scatter(data_predict[train_size:, 1], data_predict[train_size:, 0], c='red', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='Predicted Trajectory (Test)', s = 10)

# 添加颜色条
# cbar = plt.colorbar(scatter, orientation='vertical', label='Geo Altitude')

# 设置标题
plt.title('Trajectory')

# 添加图例
plt.legend()

plt.show()
