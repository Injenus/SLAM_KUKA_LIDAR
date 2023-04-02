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
cell_size = 0.1
map_size = (11, 9)
map_cell_size = tuple(map(lambda x: int(x / cell_size), map_size))
karta = np.zeros((1, map_cell_size[0], map_cell_size[1]))

log_odds_ratio = np.log(1)  # log odds ratio for unknown cells

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


def get_cell_xy(x, y):
    global cell_size
    return [int(x / cell_size), int(y / cell_size)]


odometry_data = data[0:1600, 0:3]
predict_move = np.ndarray(shape=(0, 3), dtype=float)
predict_move = np.append(predict_move, [
    [odometry_data[0][0], odometry_data[0][1], odometry_data[0][2]]], axis=0)
for i in range(1, odometry_data.shape[0]):
    w = odometry_data[i][2]
    r = distance_btw_points(odometry_data[i - 1][0], odometry_data[i - 1][1],
                            odometry_data[i][0], odometry_data[i][1])
    x = math.cos(predict_move[i - 1][2]) * r + predict_move[i - 1][0]
    y = math.sin(predict_move[i - 1][2]) * r + predict_move[i - 1][1]
    predict_move = np.append(predict_move, [[x, y, w]],
                             axis=0)


def update(frame):
    global karta, isInit
    print(frame)
    if isInit:
        isInit = False
    else:
        odo_x, odo_y, odo_w = predict_move[frame][0], predict_move[frame][1], \
                              predict_move[frame][2]
        lidar_data = data[frame][3]
        cell_robot_xy = [int(odo_x / cell_size), int(odo_y / cell_size)]
        #karta[frame][cell_robot_xy[0], cell_robot_xy[1]] = 1

        rads_lidar = np.linspace(-fov / 360 * np.pi, fov / 360 * np.pi,
                                 len(lidar_data))
        x_l = odo_x + math.cos(odo_w) * 0.3
        y_l = odo_y + math.sin(odo_w) * 0.3
        cell_lidar_xy = [int(x_l / cell_size), int(y_l / cell_size)]

        # for i in range(len(lidar_data)):
        #     x = lidar_data[i] * math.cos(odo_w - rads_lidar[i]) + x_l
        #     y = lidar_data[i] * math.sin(odo_w - rads_lidar[i]) + y_l
        #     cell_obs_x = int(x / cell_size)
        #     cell_obs_y = int(y / cell_size)
        #     if min_range <= distance_btw_points(x_l, y_l, x, y) <= max_range:
        #         try:
        #             karta[frame][cell_obs_x, cell_obs_y] = 1
        #         except IndexError:
        #             pass

        for i in range(len(lidar_data)):
            x = lidar_data[i] * math.cos(odo_w - rads_lidar[i]) + x_l
            y = lidar_data[i] * math.sin(odo_w - rads_lidar[i]) + y_l
            cell_obs_x = int(x / cell_size)
            cell_obs_y = int(y / cell_size)
            if min_range <= distance_btw_points(x_l, y_l, x, y) <= max_range:
                # Update occupancy probability using log odds ratio
                try:
                    if karta[frame][cell_obs_x, cell_obs_y] == 0.5:
                        karta[frame][cell_obs_x, cell_obs_y] = np.exp(
                            log_odds_ratio) / (1 + np.exp(log_odds_ratio))
                        print('0.5')
                    else:
                        p = karta[frame][cell_obs_x, cell_obs_y]
                        p_occ = 0.5  # probability of occupancy (tunable parameter)
                        p_free = 0.5  # probability of free space (tunable parameter)
                        z = lidar_data[i]  # sensor measurement
                        z_max = 4  # maximum sensor range
                        sigma_sq_occ = 9 ** 2  # variance of sensor model (tunable parameter)
                        sigma_sq_free = 9 ** 2  # variance of sensor model (tunable parameter)
                        p_z_occ = p_occ * np.exp(-(z - z_max) ** 2 / (
                                2 * sigma_sq_occ))  # sensor model for occupied cells
                        p_z_free = p_free * np.exp(-(z - z) ** 2 / (
                                2 * sigma_sq_free))  # sensor model for free cells
                        karta[frame][cell_obs_x, cell_obs_y] = p_z_occ / (
                                p_z_occ + p_z_free)  # Bayes' rule
                        karta[frame][cell_obs_x, cell_obs_y] = np.log(
                            karta[frame][cell_obs_x, cell_obs_y] / (
                                    1 - karta[frame][
                                cell_obs_x, cell_obs_y]))  # log odds ratio

                    # if karta[frame][cell_obs_x, cell_obs_y] > -1:
                    #     print(karta[frame][cell_obs_x, cell_obs_y])
                except IndexError:
                    pass

                # if cell_obs_x >= 0 and cell_obs_x < map_cell_size[
                #     0] and cell_obs_y >= 0 and cell_obs_y < map_cell_size[1]:
                #     if karta[frame][cell_obs_x, cell_obs_y] == 0.5:
                #         karta[frame][cell_obs_x, cell_obs_y] = np.exp(
                #             log_odds_ratio) / (1 + np.exp(log_odds_ratio))
                #     else:
                #         karta[frame][cell_obs_x, cell_obs_y] = 1.0 - np.exp(
                #             log_odds_ratio) / (1 + np.exp(log_odds_ratio))

        # x = np.multiply(lidar_data, np.cos(odo_w - rads_lidar)) + x_l
        # y = np.multiply(lidar_data, np.sin(odo_w - rads_lidar)) + y_l
        # cell_obstacle_x = np.rint(np.divide(x, cell_size)).astype(int)
        # cell_obstacle_y = np.rint(np.divide(y, cell_size)).astype(int)
        # for o in range(cell_obstacle_x.shape[0]):
        #     if min_range <= distance_btw_points(x_l, y_l, x[o],
        #                                         y[o]) <= max_range:
        #         try:
        #             karta[frame][cell_obstacle_x[o], cell_obstacle_y[o]] = 1
        #         except IndexError:
        #             pass
        karta = np.concatenate(
            (
                karta,
                np.reshape(karta[-1], (1, karta.shape[1], karta.shape[2]))),
            axis=0)
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(karta[frame], cmap='PuOr', vmin=-0.2, vmax=0.5)
    plt.imshow(karta[frame], cmap='PuOr')
    return plt


ani = animation.FuncAnimation(plt.gcf(), update, frames=100,
                              interval=1, repeat=False, blit=False)
plt.show()
