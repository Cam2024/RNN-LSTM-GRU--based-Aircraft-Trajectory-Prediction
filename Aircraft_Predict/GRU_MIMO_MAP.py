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
dataset = training_set.iloc[:, 1:].values
# 分别处理经度、纬度、地理高度的数据
latitude_data = training_set.iloc[:, 1:2].values
longitude_data = training_set.iloc[:, 2:3].values
altitude_data = training_set.iloc[:, 3:4].values

# 数据归一化
sc_long = MinMaxScaler()
sc_lat = MinMaxScaler()
sc_alt = MinMaxScaler()

longitude_data_norm = sc_long.fit_transform(longitude_data)
latitude_data_norm = sc_lat.fit_transform(latitude_data)
altitude_data_norm = sc_alt.fit_transform(altitude_data)


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
x_long, y_long = sliding_windows(longitude_data_norm, seq_length)
x_lat, y_lat = sliding_windows(latitude_data_norm, seq_length)
x_alt, y_alt = sliding_windows(altitude_data_norm, seq_length)

# 划分训练集和测试集
train_size = int(len(y_long) * 0.9)

dataX_long = torch.Tensor(x_long).cuda()
dataY_long = torch.Tensor(y_long).cuda()
trainX_long = dataX_long[:train_size]
trainY_long = dataY_long[:train_size]
testX_long = dataX_long[train_size:]
testY_long = dataY_long[train_size:]

dataX_lat = torch.Tensor(x_lat).cuda()
dataY_lat = torch.Tensor(y_lat).cuda()
trainX_lat = dataX_lat[:train_size]
trainY_lat = dataY_lat[:train_size]
testX_lat = dataX_lat[train_size:]
testY_lat = dataY_lat[train_size:]

dataX_alt = torch.Tensor(x_alt).cuda()
dataY_alt = torch.Tensor(y_alt).cuda()
trainX_alt = dataX_alt[:train_size]
trainY_alt = dataY_alt[:train_size]
testX_alt = dataX_alt[train_size:]
testY_alt = dataY_alt[train_size:]

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

input_size = 1  # 输入维度为1，即只有一个特征
hidden_size = 128  # 根据需求设定
num_layers = 2  # 减少隐藏层数量
output_size = 1  # 输出维度为1，即只有一个特征

# 实例化模型
gru_long = GRU(input_size, hidden_size, num_layers, output_size).cuda()
gru_lat = GRU(input_size, hidden_size, num_layers, output_size).cuda()
gru_alt = GRU(input_size, hidden_size, num_layers, output_size).cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_long = torch.optim.Adam(gru_long.parameters(), lr=learning_rate)
optimizer_lat = torch.optim.Adam(gru_lat.parameters(), lr=learning_rate)
optimizer_alt = torch.optim.Adam(gru_alt.parameters(), lr=learning_rate)

# 训练模型
train_losses_long = []
test_losses_long = []

train_losses_lat = []
test_losses_lat = []

train_losses_alt = []
test_losses_alt = []

for epoch in range(num_epochs):
    # 训练经度模型
    gru_long.train()
    outputs_long = gru_long(trainX_long)
    optimizer_long.zero_grad()
    loss_long = criterion(outputs_long, trainY_long)
    train_losses_long.append(loss_long.item())
    loss_long.backward()
    optimizer_long.step()

    # 训练纬度模型
    gru_lat.train()
    outputs_lat = gru_lat(trainX_lat)
    optimizer_lat.zero_grad()
    loss_lat = criterion(outputs_lat, trainY_lat)
    train_losses_lat.append(loss_lat.item())
    loss_lat.backward()
    optimizer_lat.step()

    # 训练地理高度模型
    gru_alt.train()
    outputs_alt = gru_alt(trainX_alt)
    optimizer_alt.zero_grad()
    loss_alt = criterion(outputs_alt, trainY_alt)
    train_losses_alt.append(loss_alt.item())
    loss_alt.backward()
    optimizer_alt.step()

    print("Epoch: %d, Train Loss: %1.5f" % (epoch, math.sqrt(loss_long.item())))
    # 验证集上的损失
    with torch.no_grad():
        # 经度模型验证
        gru_long.eval()
        test_outputs_long = gru_long(testX_long)
        test_loss_long = criterion(test_outputs_long, testY_long)
        test_losses_long.append(test_loss_long.item())

        # 纬度模型验证
        gru_lat.eval()
        test_outputs_lat = gru_lat(testX_lat)
        test_loss_lat = criterion(test_outputs_lat, testY_lat)
        test_losses_lat.append(test_loss_lat.item())

        # 地理高度模型验证
        gru_alt.eval()
        test_outputs_alt = gru_alt(testX_alt)
        test_loss_alt = criterion(test_outputs_alt, testY_alt)
        test_losses_alt.append(test_loss_alt.item())


# 计算RMSE并绘制图表
train_rmse_long = torch.sqrt(torch.tensor(train_losses_long))
test_rmse_long = torch.sqrt(torch.tensor(test_losses_long))

train_rmse_lat = torch.sqrt(torch.tensor(train_losses_lat))
test_rmse_lat = torch.sqrt(torch.tensor(test_losses_lat))

train_rmse_alt = torch.sqrt(torch.tensor(train_losses_alt))
test_rmse_alt = torch.sqrt(torch.tensor(test_losses_alt))

# 绘制损失图表
plt.plot(train_rmse_long, label='Longitude Train Loss')
plt.plot(test_rmse_long, label='Longitude Test Loss')
plt.plot(train_rmse_lat, label='Latitude Train Loss')
plt.plot(test_rmse_lat, label='Latitude Test Loss')
plt.plot(train_rmse_alt, label='Altitude Train Loss')
plt.plot(test_rmse_alt, label='Altitude Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Root Mean Squared Error (RMSE)')
plt.legend()
plt.show()

# 模型评估
gru_long.eval()
train_predict_long = gru_long(dataX_long).cpu().data.numpy()
data_predict_long = sc_long.inverse_transform(train_predict_long)
dataY_plot_long = sc_long.inverse_transform(dataY_long.cpu().data.numpy())

gru_lat.eval()
train_predict_lat = gru_lat(dataX_lat).cpu().data.numpy()
data_predict_lat = sc_lat.inverse_transform(train_predict_lat)
dataY_plot_lat = sc_lat.inverse_transform(dataY_lat.cpu().data.numpy())

gru_alt.eval()
train_predict_alt = gru_alt(dataX_alt).cpu().data.numpy()
data_predict_alt = sc_alt.inverse_transform(train_predict_alt)
dataY_plot_alt = sc_alt.inverse_transform(dataY_alt.cpu().data.numpy())

# 合并三个维度
data_predict = np.concatenate((data_predict_long, data_predict_lat, data_predict_alt), axis=-1)
dataY_plot = np.concatenate((dataY_plot_long, dataY_plot_lat, dataY_plot_alt), axis=1)

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
scatter = ax.scatter(dataset[:, 1], dataset[:, 0], c='blue', cmap='viridis', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='True Trajectory', s = 10)

# 绘制预测轨迹
ax.scatter(data_predict[:train_size, 0], data_predict[:train_size, 1], c='green', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='Predicted Trajectory (Train)', s = 10)
ax.scatter(data_predict[train_size:, 0], data_predict[train_size:, 1], c='red', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(), label='Predicted Trajectory (Test)', s = 10)

# 添加颜色条
# cbar = plt.colorbar(scatter, orientation='vertical', label='Geo Altitude')

# 设置标题
plt.title('Trajectory')

# 添加图例
plt.legend()

plt.show()

