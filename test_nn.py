import numpy as np
import matplotlib.pyplot as plt

# Define the LIDAR sensor parameters
num_beams = 511
fov = np.pi
angle_incr = fov / num_beams
min_range = 0.05
max_range = 4



# Define the robot motion model
def motion_model(x, u):
    # x = [x, y, theta]
    # u = [v, w]
    v = u[0]
    w = u[1]
    theta = x[2]
    dt = 1.0
    x_next = np.array([
        x[0] + v * np.cos(theta) * dt,
        x[1] + v * np.sin(theta) * dt,
        theta + w * dt
    ])
    return x_next


# Define the LIDAR sensor model
def sensor_model(x, lidar_data):
    # x = [x, y, theta]
    # lidar_data = [d1, d2, ..., dn]
    ranges = []
    for i in range(num_beams):
        angle = i * angle_incr - fov / 2
        r = max_range
        for d in lidar_data:
            if d < r:
                r = d
        if r < min_range:
            r = min_range
        if r > max_range:
            r = max_range
        ranges.append(r)
    return ranges


# Define the occupancy grid mapping function
def occupancy_grid_mapping(x, ranges, map_size, cell_size):
    # x = [x, y, theta]
    # ranges = [r1, r2, ..., rn]
    # map_size = [width, height]
    # cell_size = size of each cell in meters
    map_width, map_height = map_size
    cell_width = int(map_width / cell_size)
    cell_height = int(map_height / cell_size)
    map = np.zeros((cell_width, cell_height))
    for i in range(num_beams):
        angle = i * angle_incr - fov / 2 + x[2]
        dx = np.cos(angle) * ranges[i]
        dy = np.sin(angle) * ranges[i]
        x_i = int((x[0] + dx) / cell_size)
        y_i = int((x[1] + dy) / cell_size)
        if x_i >= 0 and x_i < cell_width and y_i >= 0 and y_i < cell_height:
            map[x_i][y_i] = 1
    return map


# Define the SLAM algorithm
def slam(data, map_size, cell_size):
    # Initialize the robot pose
    x = np.array([data[0][0], data[0][1], data[0][2]])

    # Initialize the occupancy grid karta
    map = np.zeros((int(map_size[0]/cell_size), int(map_size[1]/cell_size)))

    # Loop over the data
    for i in range(len(data)):
        # Get the robot pose and lidar data
        x_i = np.array([data[i][0], data[i][1], data[i][2]])
        lidar_data = data[i][3]

        # Update the robot pose using odometry
        u = np.array([np.linalg.norm(x_i[:2] - x[:2]), x_i[2] - x[2]])
        x = motion_model(x, u)

        # Update the occupancy grid karta using LIDAR data
        ranges = sensor_model(x, lidar_data)
        map += occupancy_grid_mapping(x, ranges, map_size, cell_size)



    return map



# # Example usage
# data = [[1.0, 2.0, np.pi / 4, [1.0, 2.0, 3.0, 2.5, 1.2, 1.0, 1.0, 1.5]],
#         [3.0, 4.0, np.pi / 2, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]],
#         [5.0, 6.0, 3 * np.pi / 4, [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
#         [7.0, 8.0, np.pi, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]
#         ]

file_name = 'all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)

map_size = (100, 100)
cell_size = 0.1
map = slam(data, map_size, cell_size)
print(map)

plt.imshow(map)
plt.show()