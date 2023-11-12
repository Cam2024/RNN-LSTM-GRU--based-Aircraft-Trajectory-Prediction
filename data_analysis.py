import pandas as pd
import matplotlib.pyplot as plt
import time

# 读取CSV文件
data = pd.read_csv('sorted_1.csv')  # 请将'your_data.csv'替换为您的CSV文件路径

num = 848

def plot_flight(num):
    # 提取aircraft为2的飞机数据
    flight_data = data[data['aircraft'] == num]
    if flight_data.empty:
        return
    # 提取latitude、longitude和geoAltitude信息
    latitude = flight_data['latitude']
    longitude = flight_data['longitude']
    geoAltitude = flight_data['geoAltitude']

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞行轨迹
    ax.plot(latitude, longitude, geoAltitude, label = f'Flight Path: {num}', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Geo Altitude')

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()


plot_flight(num)