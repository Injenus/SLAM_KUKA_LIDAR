import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from PIL import Image

res_file_name = 'all_data.csv'
direc = 'frame by frame/'
lidar_angle = 240  # угол обзора лидара в градусах
threshold = 0.125  # сколько обзора в доле с каждого края закрывается самим
# роботом (положим, что лидар устновлен симметрично)
max_range = 4.0  # предел лидара
enable_frames = True  # разрешение на запись промежуточных кадров
dpi = 100

data_path = 'data/'
df = pd.DataFrame(columns=['X', 'Y', 'W', 'Lidar_Data'])
for filename in os.listdir(data_path):
    with open(os.path.join(data_path, filename), 'r') as f:
        for line in f:
            readed_line = line
            curr_data = [readed_line[:-1].split(';')[0].split(','),
                         readed_line[:-1].split(';')[1].split(
                             ',')]  # [0] - odo, [1] - lidar
            parsed_data = []
            for i, l in enumerate(curr_data):
                # print(l,i)
                if i == 0:
                    for j, el in enumerate(l):
                        parsed_data.append(float(el))
                else:
                    parsed_data.append([float(item) for item in l])
            # в итоге 4 ячейки: float, float, float, list
            df.loc[len(df)] = parsed_data
df.to_csv(res_file_name, header=True, sep=';', index=False)


def del_inf(a, d):  # удаляет элементы массива а по индексам списка d
    return np.delete(a, d)


def create_by_index(old, mask):  # новый массив по индексам из старого __ 1D
    b = np.array([], float)
    for ind in mask:
        b = np.append(b, [old[ind]])
    return b


g_center_bot = [df['X'].sum() / df.shape[0], df['Y'].sum() / df.shape[0]]
map_xy = np.ndarray(shape=(0, 2), dtype=float)
robot_xy = np.ndarray(shape=(0, 2), dtype=float)
lidar_xy = np.ndarray(shape=(0, 2), dtype=float)
inf_xy = np.ndarray(shape=(0, 2), dtype=float)
interfer_xy = np.ndarray(shape=(0, 2), dtype=float)
laser_xy = np.ndarray(shape=(0, 2), dtype=float)

for i in range(df.shape[0]):
    rads_lidar = np.linspace(-lidar_angle / 360 * np.pi,
                             lidar_angle / 360 * np.pi,
                             len(list(df.iloc[i]['Lidar_Data'])))
    x_l = df.iloc[i]['X'] + math.cos(df.iloc[i]['W']) * 0.3
    y_l = df.iloc[i]['Y'] + math.sin(df.iloc[i]['W']) * 0.3
    robot_xy = np.append(robot_xy, [[df.iloc[i]['X'], df.iloc[i]['Y']]],
                         axis=0)
    lidar_xy = np.append(lidar_xy, [[x_l, y_l]], axis=0)

    x = np.multiply(np.asarray(list(df.iloc[i]['Lidar_Data'])),
                    np.cos(df.iloc[i]['W'] - rads_lidar)) + x_l
    y = np.multiply(np.asarray(list(df.iloc[i]['Lidar_Data'])),
                    np.sin(df.iloc[i]['W'] - rads_lidar)) + y_l

    for k1 in range(len(x)):
        laser_xy = np.append(laser_xy, [[x[k1], y[k1]]], axis=0)

    false_obstacles_indexes = []
    Interferences = []
    for j, el in enumerate(list(df.iloc[i]['Lidar_Data'])):
        if el >= max_range:  # or el <= 0.02
            false_obstacles_indexes.append(j)
        elif 1 and (
                j < threshold * len(list(df.iloc[i]['Lidar_Data'])) or j > (
                1 - threshold) * len(list(df.iloc[i]['Lidar_Data']))):
            Interferences.append(j)

    x_inf = create_by_index(x, false_obstacles_indexes)
    y_inf = create_by_index(y, false_obstacles_indexes)
    for k2 in range(len(x_inf)):
        inf_xy = np.append(inf_xy, [[x_inf[k2], y_inf[k2]]],
                           axis=0)

    x_body = create_by_index(x, Interferences)
    y_body = create_by_index(y, Interferences)
    for k3 in range(len(x_body)):
        interfer_xy = np.append(interfer_xy, [[x_body[k3], y_body[k3]]],
                                axis=0)

    x_map = del_inf(x, false_obstacles_indexes + Interferences)
    y_map = del_inf(y, false_obstacles_indexes + Interferences)
    for k4 in range(len(x_map)):
        map_xy = np.append(map_xy, [[x_map[k4], y_map[k4]]], axis=0)

    if enable_frames:
        if not os.path.exists(direc):
            os.makedirs(direc)

        # plt.figure('Data from the loop{}'.format(i))
        plt.clf()
        plt.title('Data from the loop{}'.format(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_map, y_map, '.', color='m', label='Map')
        plt.plot(x_inf, y_inf, '.', color='y', label='INF')
        plt.plot(x_body, y_body, '.', color='chocolate', label='Interferences')
        plt.plot(x, y, ':', color='c', label='Laser')
        plt.plot(x_l, y_l, '.', color='r', label='Lidar')  # лидар
        plt.plot(df.iloc[i]['X'], df.iloc[i]['Y'], '.', color='k',
                 label='Robot')  # робот
        # plt.legend()
        plt.xlim(g_center_bot[0] - 7, g_center_bot[0] + 7)
        plt.ylim(g_center_bot[1] - 7, g_center_bot[1] + 7)
        plt.grid()
        plt.savefig(direc + '{}_{}.jpg'.format(res_file_name[:-4], i),
                    dpi=dpi, bbox_inches='tight')
    if (i + 1) / df.shape[0] * 100 % 5 == 0: print(
        '{}% completed'.format((i + 1) / df.shape[0] * 100))

frames = []
for i in range(df.shape[0]):
    frame = Image.open(direc + '{}_{}.jpg'.format(res_file_name[:-4], i))
    frames.append(frame)
frames[0].save(
    direc + 'MOVE.gif',
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=150,
    loop=0
)

g_center_raw = np.sum(laser_xy, axis=0) / len(laser_xy)

plt.figure('Data from the Memory')
plt.title('According to the "{}"'.format(res_file_name))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(map_xy[:, 0], map_xy[:, 1], '.', color='m', label='Map', zorder=10)
plt.plot(inf_xy[:, 0], inf_xy[:, 1], '.', color='y', label='INF')

plt.plot(laser_xy[:, 0], laser_xy[:, 1], ':', color='c', label='Laser')
plt.plot(interfer_xy[:, 0], interfer_xy[:, 1], '.', color='chocolate',
         label='Interferences')
plt.plot(lidar_xy[:, 0], lidar_xy[:, 1], '.', color='r',
         label='Lidar')  # лидар
plt.plot(robot_xy[:, 0], robot_xy[:, 1], '.', color='k',
         label='Robot')  # робот
plt.legend()
plt.xlim(g_center_raw[0] - 8, g_center_raw[0] + 8)
plt.ylim(g_center_raw[1] - 8, g_center_raw[1] + 8)
plt.grid()
if not os.path.exists(direc):
    os.makedirs(direc)
plt.savefig(direc + 'ALL.jpg', dpi=dpi, bbox_inches='tight')

plt.figure('MAP')
plt.xticks([])
plt.yticks([])
plt.scatter(map_xy[:, 0], map_xy[:, 1], s=0.1, color='k')
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
plt.xlim(g_center_raw[0] - 8, g_center_raw[0] + 8)
plt.ylim(g_center_raw[1] - 8, g_center_raw[1] + 8)
plt.savefig(direc + 'Map_RAW.jpg', dpi=dpi, bbox_inches='tight')

plt.show()
