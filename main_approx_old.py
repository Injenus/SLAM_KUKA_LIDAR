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
map_size = (13, 11)
map_cell_size = tuple(map(lambda x: int(x / cell_size), map_size))
karta = np.zeros((1, map_cell_size[0], map_cell_size[1]))
karta -= 1
fov = 180
max_range = 4
min_range = 0.1
# оффсет нужно вычитать из одометрии
offset = (-1.3, 3.9)
for i in range(data.shape[0]):
    data[i][0] -= offset[0]
    data[i][1] -= offset[1]
isInit = True


def distance_btw_points(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_cell_xy(x, y):
    global cell_size
    return [int(x / cell_size), int(y / cell_size)]


def get_coord_xy(cell_x, cell_y):
    global cell_size
    return [cell_x * cell_size + np.sign(cell_x) * cell_size / 2,
            cell_y * cell_size + np.sign(cell_y) * cell_size / 2]


def eq_line(x1, y1, x2, y2):
    # xa+yb+c=0      a,b,c
    return [y2 - y1, -x2 - x1, y1 * x2 - x1 * y2]


print(data[0][0], data[0][1], data[0][2])
print(get_cell_xy(data[0][0], data[0][1]))
print(get_coord_xy(get_cell_xy(data[0][0], data[0][1])[0],
                   get_cell_xy(data[0][0], data[0][1])[1]))


def init(frame):
    global karta, obstacles, ref_obss
    odo_x, odo_y, odo_w = data[frame][0], data[frame][1], data[frame][2]
    lidar_data = data[frame][3]
    cell_robot_xy = get_cell_xy(odo_x, odo_y)
    karta[frame][cell_robot_xy[0]][cell_robot_xy[1]] = 0.5

    rads_lidar = np.linspace(-fov / 360 * np.pi, fov / 360 * np.pi,
                             len(lidar_data))
    x_l = odo_x + math.cos(odo_w) * 0.3
    y_l = odo_y + math.sin(odo_w) * 0.3
    cell_lidar_xy = get_cell_xy(x_l, y_l)
    karta[frame][cell_lidar_xy[0]][cell_lidar_xy[1]] = 0.7

    # x_m = odo_x + math.cos(odo_w +1.57) * 0.3
    # y_m = odo_y + math.sin(odo_w+1.57) * 0.3
    # cell_m_xy = get_cell_xy(x_m, y_m)
    # karta[frame][cell_m_xy[0]][cell_m_xy[1]] = 0.7
    # line_params = eq_line(odo_x, odo_y, x_m, y_m)
    # print(line_params)

    x = np.multiply(lidar_data, np.cos(odo_w - rads_lidar)) + x_l
    y = np.multiply(lidar_data, np.sin(odo_w - rads_lidar)) + y_l
    cell_obstacle_x = np.rint(np.divide(x, cell_size)).astype(int)
    cell_obstacle_y = np.rint(np.divide(y, cell_size)).astype(int)
    for o in range(cell_obstacle_x.shape[0]):
        if min_range <= distance_btw_points(x_l, y_l, x[o],
                                            y[o]) <= max_range:
            try:
                karta[frame][
                    cell_obstacle_x[o], cell_obstacle_y[o]] = 1
            except IndexError:
                pass
    # (x-x_l)**2+(y-y_l)**2<=r**2
    dw = 1
    for i in range(len(karta[frame])):
        for j in range(len(karta[frame][i])):
            coord_xy = get_coord_xy(i, -j)
            if ((coord_xy[0] - x_l) ** 2 + (
                    coord_xy[1] - y_l) ** 2) <= 4 ** 2 and (
                    odo_w - dw < math.atan2(coord_xy[1] - y_l, coord_xy[
                                                                   0] - x_l) < odo_w + dw or -3.14 < math.atan2(
                coord_xy[1] - y_l, coord_xy[0] - x_l) < -3.14 + (
                            dw - (3.14 - odo_w))):
                if karta[frame][i][-j] == -1:
                    karta[frame][i][-j] = 0
                # elif karta[frame][i][j] == 1:
                #     id = len(obstacles)
                #     obstacles = np.append(obstacles, [
                #         [id, coord_xy[0], coord_xy[1]]],
                #                           axis=0)
                #     ref_obss.append(id)

    karta = np.concatenate(
        (
            karta,
            np.reshape(karta[-1], (1, karta.shape[1], karta.shape[2]))),
        axis=0)


def cycle(frame):
    pass


def update(frame):
    global karta, isInit
    print(frame)
    if isInit:
        isInit = False
        init(frame)

    else:

        pass

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(karta[frame], cmap='PuOr', vmin=-0.2, vmax=1.1)
    return plt


ani = animation.FuncAnimation(plt.gcf(), update, frames=1,  # data.shape[0],
                              interval=1, repeat=False, blit=False)
plt.show()
