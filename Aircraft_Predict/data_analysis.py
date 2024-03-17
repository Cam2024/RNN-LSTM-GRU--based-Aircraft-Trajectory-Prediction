import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 从CSV文件中读取数据
data_df = pd.read_csv('dataset_3.csv')

# 创建地图
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置地图的范围为整个欧洲
ax.set_extent([-12, 40, 35, 70], crs=ccrs.PlateCarree())

# 添加地图特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgrey')

# 绘制经纬度轨迹
scatter = ax.scatter(data_df['longitude'], data_df['latitude'], c=data_df['geoAltitude'], cmap='viridis', alpha=0.8, edgecolors='none', transform=ccrs.PlateCarree(),s=5)

# 添加颜色条
cbar = plt.colorbar(scatter, orientation='vertical', label='Geo Altitude')

# 设置标题
plt.title('Trajectory with Geo Altitude')

plt.show()
