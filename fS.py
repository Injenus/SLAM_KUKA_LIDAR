import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
from numpy.random import randn
from scipy.stats import multivariate_normal

matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1

file_name = 'all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)
odom_data = data[0:1600, 0:3]
lidar_data = data[0:1600, 3:4]


def motion_model(x, u, dt):
    """
    Motion model for the robot, which updates the robot's pose based on the control input (u) and elapsed time (dt).
    The control input u consists of [x_dot, y_dot, yaw_rate], which represent the linear and angular velocities of the robot.
    The robot's pose is represented by [x, y, theta], which represent the robot's position and orientation in the world frame.
    """
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[2] = normalize_angle(x[2])
    return x