import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, cos, remainder, tau, atan2

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
fov = 180
max_range = 4
min_range = 0.1
# оффсет нужно вычитать из одометрии
offset = (-0.3, 2.7)
for i in range(data.shape[0]):
    data[i][0] -= offset[0]
    data[i][1] -= offset[1]
isInit = True

############ GLOBAL VARIABLES ###################
state_pub = None
############## NEEDED BY EKF ####################
# Process and sensing noise covariances.
V = np.array([[0.02 ** 2, 0.0], [0.0, (np.pi / 360) ** 2]])
W = np.array([[0.1 ** 2, 0.0], [0.0, (np.pi / 180) ** 2]])
# Initial vehicle state mean and covariance.
x_t = None
P_t = np.array([[0.01**2,0.0,0.0],[0.0,0.01**2,0.0],[0.0,0.0,0.005**2]])
# Most recent odom reading and landmark measurements.
odom_queue = []; lm_meas_queue = []
# IDs of seen landmarks. Order corresponds to ind in state.
lm_IDs = []
# current timestep number.
timestep = 0
#################################################


def distance_btw_points(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def ekf(event):
    global x_t, P_t, lm_IDs, lm_meas_queue, odom_queue, timestep
    # skip if params not read yet or there's no prediction or measurement.
    if W is None or x_t is None or len(odom_queue) < 1 or len(
            lm_meas_queue) < 1:
        return
    timestep += 1
    # pop the next data off the queue.
    odom = odom_queue.pop(0)
    lm_meas = lm_meas_queue.pop(0)

    ############# PREDICTION STEP ##################
    # odom gives us a (dist, heading) "command".
    d_d = odom[0]
    d_th = odom[1]

    # Compute jacobian matrices.
    F_xv = np.array([[1, 0, -d_d * sin(x_t[2, 0])],
                     [0, 1, d_d * cos(x_t[2, 0])],
                     [0, 0, 1]])
    F_x = np.eye(x_t.shape[0])  # * 0
    # F_x = np.zeros((x_t.shape[0],x_t.shape[0]))
    F_x[0:3, 0:3] = F_xv
    F_vv = np.array([[cos(x_t[2, 0]), 0],
                     [sin(x_t[2, 0]), 0],
                     [0, 1]])
    F_v = np.zeros((x_t.shape[0], 2))
    F_v[0:3, 0:2] = F_vv

    # Make predictions.
    # landmarks are assumed constant, so we predict only vehicle position will change.
    x_pred = x_t
    x_pred[0, 0] = x_t[0, 0] + (d_d + 0) * cos(x_t[2, 0])
    x_pred[1, 0] = x_t[1, 0] + (d_d + 0) * sin(x_t[2, 0])
    x_pred[2, 0] = x_t[2, 0] + d_th + 0
    # cap heading to (-pi,pi).
    x_pred[2, 0] = remainder(x_pred[2, 0], tau)
    # propagate covariance.
    P_pred = F_x @ P_t @ F_x.T + F_v @ V @ F_v.T

    ################## UPDATE STEP #######################
    # we only use landmark measurements, so skip if there aren't any.
    if len(lm_meas) > 0:
        # at least one landmark was detected since the last EKF iteration.
        # we can run the update step once for each landmark.
        num_landmarks = len(lm_meas) // 3
        for l in range(num_landmarks):
            # extract single landmark observation.
            id = int(lm_meas[l * 3]);
            r = lm_meas[l * 3 + 1];
            b = lm_meas[l * 3 + 2]
            # check if this is the first detection of this landmark ID.
            if id in lm_IDs:
                ################ LANDMARK UPDATE ###################
                # this landmark is already in our state, so update it.
                i = lm_IDs.index(id) * 2 + 3  # index of lm x in state.

                # Compute Jacobian matrices.
                dist = ((x_t[i, 0] - x_pred[0, 0]) ** 2 + (
                            x_t[i + 1, 0] - x_pred[1, 0]) ** 2) ** (1 / 2)
                H_xv = np.array([[-(x_t[i, 0] - x_pred[0, 0]) / dist,
                                  -(x_t[i + 1, 0] - x_pred[1, 0]) / dist, 0],
                                 [(x_t[i + 1, 0] - x_pred[1, 0]) / (dist ** 2),
                                  -(x_t[i, 0] - x_pred[0, 0]) / (dist ** 2),
                                  -1]])
                H_xp = np.array([[(x_t[i, 0] - x_pred[0, 0]) / dist,
                                  (x_t[i + 1, 0] - x_pred[1, 0]) / dist], [
                                     -(x_t[i + 1, 0] - x_pred[1, 0]) / (
                                                 dist ** 2),
                                     (x_t[i, 0] - x_pred[0, 0]) / (
                                                 dist ** 2)]])
                H_x = np.zeros((2, x_pred.shape[0]))
                H_x[0:2, 0:3] = H_xv
                H_x[0:2, i:i + 2] = H_xp
                H_w = np.eye(2)

                # Update the state and covariance.
                # compute innovation.
                ang = remainder(atan2(x_t[i + 1, 0] - x_pred[1, 0],
                                      x_t[i, 0] - x_pred[0, 0]) - x_pred[2, 0],
                                tau)
                z_est = np.array([[dist], [ang]])
                nu = np.array([[r], [b]]) - z_est - np.array(
                    [[params["w_r"]], [params["w_b"]]])
                # compute kalman gain.
                S = H_x @ P_pred @ H_x.T + H_w @ W @ H_w.T
                K = P_pred @ H_x.T @ np.linalg.inv(S)
                # perform update.
                x_pred = x_pred + K @ nu
                # cap heading to (-pi,pi).
                x_pred[2, 0] = remainder(x_pred[2, 0], tau)
                P_pred = P_pred - K @ H_x @ P_pred
            else:
                ############# LANDMARK INSERTION #######################
                # this is our first time detecting this landmark ID.
                n = x_pred.shape[0]
                # add the new landmark to our state.
                g = np.array([[x_pred[0, 0] + r * cos(x_pred[2, 0] + b)],
                              [x_pred[1, 0] + r * sin(x_pred[2, 0] + b)]])
                x_pred = np.vstack([x_pred, g])
                # add landmark ID to our list.
                lm_IDs.append(id)
                # Compute Jacobian matrices.
                G_z = np.array(
                    [[cos(x_pred[2, 0] + b), -r * sin(x_pred[2, 0] + b)],
                     [sin(x_pred[2, 0] + b), r * cos(x_pred[2, 0] + b)]])
                G_x = np.array([[1, 0, -r * sin(x_pred[2, 0] + b)],
                                [0, 1, r * cos(x_pred[2, 0] + b)]])
                # form insertion jacobian.
                Y = np.eye(n + 2)
                Y[n:n + 2, n:n + 2] = G_z
                Y[n:n + 2, 0:3] = G_x

                # update covariance.
                P_pred = Y @ np.vstack([np.hstack([P_pred, np.zeros((n, 2))]),
                                        np.hstack(
                                            [np.zeros((2, n)), W])]) @ Y.T

    # finalize the state update. this works even if no landmarks were detected.
    x_t = x_pred
    P_t = P_pred
    # cap heading to (-pi,pi).
    x_t[2, 0] = remainder(x_t[2, 0], tau)


def predict():
    pass


def update():
    pass


def anim(frame):
    global karta, isInit
    print(frame)
    if isInit:
        isInit = False
    else:
        # karta[np.random.randint(0, map_cell_size[0]), np.random.randint(0,
        #                                                               map_cell_size[
        #                                                                   1])] = 0.5
        odo_x, odo_y, odo_w = data[frame][0], data[frame][1], data[frame][2]
        lidar_data = data[frame][3]
        cell_robot_xy = [int(odo_x / cell_size), int(odo_y / cell_size)]
        karta[frame][cell_robot_xy[0], cell_robot_xy[1]] = 0.5

        rads_lidar = np.linspace(-fov / 360 * np.pi, fov / 360 * np.pi,
                                 len(lidar_data))
        x_l = odo_x + cos(odo_w) * 0.3
        y_l = odo_y + sin(odo_w) * 0.3
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


ani = animation.FuncAnimation(plt.gcf(), anim, frames=data.shape[0],
                              interval=1, repeat=False, blit=False)
plt.show()
