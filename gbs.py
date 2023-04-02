import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Define grid parameters
cell_size = 0.1  # meters
grid_size = 2000  # cells
grid_origin = np.array([-70., -70.])  # meters

# Initialize occupancy grid
occupancy_grid = np.zeros((grid_size, grid_size))
log_odds_ratio = np.log(0.5/0.5)  # log odds ratio for unknown cells

# Load odometry and LIDAR data
file_name = 'all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)

odometry_data = data[0:1600, 0:3]
lidar_data = data[:, 3]

# Initialize robot pose
robot_pose = np.array([odometry_data[0][0], odometry_data[0][1],
                       odometry_data[0][2]])  # x, y, theta
isInit = True


def update(i):
    global isInit
    if isInit:
        isInit = False
    print(i)
    # Loop over data
    # for i in range(len(odometry_data)):
    # Update robot pose based on odometry data
    delta_x = odometry_data[i + 1][0] - odometry_data[i][0]
    delta_y = odometry_data[i + 1][1] - odometry_data[i][1]
    delta_theta = odometry_data[i + 1][2] - odometry_data[i][2]
    robot_pose[0] += delta_x * np.cos(robot_pose[2]) - delta_y * np.sin(
        robot_pose[2])
    robot_pose[1] += delta_x * np.sin(robot_pose[2]) + delta_y * np.cos(
        robot_pose[2])
    robot_pose[2] += delta_theta

    # Update occupancy grid based on LIDAR data
    lidar_scan = lidar_data[i]
    angles = np.arange(-np.pi / 2, np.pi / 2, np.pi / len(lidar_scan))
    for j in range(len(lidar_scan)):
        if lidar_scan[j] < 4:  # threshold for valid measurements
            # Calculate position of cell in grid
            x = robot_pose[0] + lidar_scan[j] * np.cos(
                -angles[j] + robot_pose[2]) + math.cos(robot_pose[2]) * 0.3
            y = robot_pose[1] + lidar_scan[j] * np.sin(
                -angles[j] + robot_pose[2]) + math.sin(robot_pose[2]) * 0.3
            cell_x = int((x - grid_origin[0]) / cell_size)
            cell_y = int((y - grid_origin[1]) / cell_size)

            # Update occupancy probability using log odds ratio
            if cell_x >= 0 and cell_x < grid_size and cell_y >= 0 and cell_y < grid_size:
                if occupancy_grid[cell_y, cell_x] == 0.5:
                    occupancy_grid[cell_y, cell_x] = np.exp(log_odds_ratio) / (
                            1 + np.exp(log_odds_ratio))
                else:
                    occupancy_grid[cell_y, cell_x] = 1.0 - np.exp(
                        log_odds_ratio) / (1 + np.exp(log_odds_ratio))

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(occupancy_grid, cmap='PuOr', vmin=-0.2, vmax=1.1)
    return plt


ani = animation.FuncAnimation(plt.gcf(), update, frames=200,  # data.shape[0],
                              interval=1, repeat=False, blit=False)
plt.show()
# Save occupancy grid to file
np.save('occupancy_grid.npy', occupancy_grid)
