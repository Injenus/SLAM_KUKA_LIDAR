import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

file_name = 'D:\\PyProjects\\SLAM_KUKA_LIDAR\\all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)
"""
data = [ [x,y,w,[lidar_data_list]
                        ...
                        ]
that is data[i]=[x_i, y_i, w_i,[lidar_data_list_i]
"""
cell_size = 0.1
map_size = (11, 9)
map_cell_size = tuple(map(lambda x: int(x / cell_size), map_size))
karta = np.zeros((1, map_cell_size[0], map_cell_size[1]))


print(karta, karta.shape)

add=np.reshape(karta[-1], karta.shape)

print(add, add.shape)

karta = np.concatenate((karta, add),axis=0)
print(karta, karta.shape)
karta = np.concatenate((karta, add),axis=0)
print(karta, karta.shape)

plt.imshow(karta[2], cmap='PuOr', vmin=-0.2, vmax=1.1)
plt.show()