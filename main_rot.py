import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1

file_name = 'all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)
"""
data = [ [x,y,w,[lidar_data_list]
                        ...
                        ]
that is data[i]=[x_i, y_i, w_i,[lidar_data_list_i]
"""
cell_size = 0.035
map_size = (11, 9)
map_cell_size = tuple(map(lambda x: int(x / cell_size), map_size))
karta = np.zeros((1, map_cell_size[0], map_cell_size[1]))
fov = 180
max_range = 4
min_range = 0.1
# оффсет нужно вычитать из одометрии
offset = (-0.3, 2.7)
for i in range(data.shape[0]):
    data[i][0] -= offset[0]
    data[i][1] -= offset[1]
isInit = True


def distance_btw_points(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


c_mass_xy = []
for i in range(16):
    x = np.sum(data[:, 0][i * 100:(i + 1) * 100], axis=0) / 100
    y = np.sum(data[:, 1][i * 100:(i + 1) * 100], axis=0) / 100
    c_mass_xy.append([x, y])
rot_angs = [0 for x in range(16)]
rot_angs[1]=-0.45

xy_new = np.ndarray(shape=(0, 2), dtype=float)
for i in range(16):
    a = rot_angs[i]
    rot_matrix = np.array([[math.cos(-a), -math.sin(-a)],
                           [math.sin(-a), math.cos(-a)]])
    m_x = data[i * 100:(i + 1) * 100, 0:1] - c_mass_xy[i][0]
    m_y = data[i * 100:(i + 1) * 100, 1:2] - c_mass_xy[i][1]
    m = np.column_stack([m_x, m_y])
    m_r = np.dot(m, rot_matrix)
    m_xr = m_r[:, 0] + c_mass_xy[i][0]
    m_yr = m_r[:, 1] + c_mass_xy[i][1]
    m_r_sh = np.column_stack([m_xr, m_yr])
    xy_new = np.vstack([xy_new, m_r_sh])

print(xy_new.shape)


def update(frame):
    global karta, isInit
    print(frame)
    if isInit:
        isInit = False
    else:
        # karta[np.random.randint(0, map_cell_size[0]), np.random.randint(0,
        #                                                               map_cell_size[
        #                                                                   1])] = 0.5
        odo_x, odo_y, odo_w = xy_new[frame][0], xy_new[frame][1], data[frame][2]
        lidar_data = data[frame][3]
        cell_robot_xy = [int(odo_x / cell_size), int(odo_y / cell_size)]
        karta[frame][cell_robot_xy[0], cell_robot_xy[1]] = 0.5

        rads_lidar = np.linspace(-fov / 360 * np.pi, fov / 360 * np.pi,
                                 len(lidar_data))
        x_l = odo_x + math.cos(odo_w) * 0.3
        y_l = odo_y + math.sin(odo_w) * 0.3
        cell_lidar_xy = [int(x_l / cell_size), int(y_l / cell_size)]

        x = np.multiply(lidar_data, np.cos(odo_w - rads_lidar)) + x_l
        y = np.multiply(lidar_data, np.sin(odo_w - rads_lidar)) + y_l
        cell_obstacle_x = np.rint(np.divide(x, cell_size)).astype(int)
        cell_obstacle_y = np.rint(np.divide(y, cell_size)).astype(int)
        for o in range(cell_obstacle_x.shape[0]):
            if min_range <= distance_btw_points(x_l, y_l, x[o],
                                                y[o]) <= max_range:
                try:
                    karta[frame][cell_obstacle_x[o], cell_obstacle_y[o]] = 1
                except IndexError:
                    pass
        karta = np.concatenate(
            (
                karta,
                np.reshape(karta[-1], (1, karta.shape[1], karta.shape[2]))),
            axis=0)
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(karta[frame], cmap='PuOr', vmin=-0.2, vmax=1.1)
    return plt

ani = animation.FuncAnimation(plt.gcf(), update, frames=200,
                              interval=1, repeat=False, blit=False)
plt.show()
