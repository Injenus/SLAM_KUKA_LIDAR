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

odometry_data = data[0:1600, 0:3]
predict_move = np.ndarray(shape=(0, 3), dtype=float)
predict_move = np.append(predict_move, [
    [odometry_data[0][0], odometry_data[0][1], odometry_data[0][2]]], axis=0)


def distance_btw_points(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


for i in range(1, odometry_data.shape[0]):
    r = distance_btw_points(odometry_data[i - 1][0], odometry_data[i - 1][1],
                            odometry_data[i][0], odometry_data[i][1])
    x = math.cos(predict_move[i - 1][2]) * r + predict_move[i - 1][0]
    y = math.sin(predict_move[i - 1][2]) * r + predict_move[i - 1][1]
    predict_move = np.append(predict_move, [[x, y, odometry_data[i][2]]],
                             axis=0)

print(predict_move.shape)
