import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('sorted_dataset.csv')  # 请将'your_data.csv'替换为您的CSV文件路径

temp_num = 2
current_temp_num = temp_num


def plot_flight(temp_num):
    # 提取aircraft数据
    flight_data = data[data['aircraft'] == temp_num]

    # 提取latitude、longitude和geoAltitude信息
    latitude = flight_data['latitude']
    longitude = flight_data['longitude']
    geoAltitude = flight_data['geoAltitude']

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞行轨迹
    ax.plot(latitude, longitude, geoAltitude, label=f'Flight Path: {temp_num}', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Geo Altitude')

    # 显示图例
    ax.legend()

    plt.show()


# 启用交互模式
plt.ion()

while True:
    key = input("Press Enter to plot the next flight path or 'q' to quit: ")

    if key == 'q':
        break

    current_temp_num += 1
    plot_flight(current_temp_num)
    plt.pause(0.001)

# 关闭交互模式
plt.ioff()
plt.show()
