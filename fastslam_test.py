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

N = 200  # number of particles
particles = np.zeros((N, 3))  # [x, y, theta]
weights = np.ones(N) / N  # initialize with uniform weights

map_size = 100
map_resolution = 0.1
grid_size = int(map_size / map_resolution)
grid_map = np.zeros((grid_size, grid_size))

# odom_data = np.load('odom_data.npy')
# lidar_data = np.load('lidar_data.npy')

odom_data = data[0:1600, 0:3]
lidar_data = data[0:1600, 3:4]

Q = np.diag([0.1, 0.1]) ** 2  # process noise
R = np.diag([0.1, np.deg2rad(2)]) ** 2  # sensor noise


def sensor_model(x, z, map):
    """
    Computes the likelihood of a particle given a range-bearing measurement (z) and the map of landmarks.
    The state of the particle is represented by (x), and the map is represented as a list of landmark positions.
    """
    # Compute the expected measurement for each landmark in the map
    expected_z = []
    for i in range(len(map)):
        delta = map[i] - x[:2]
        expected_z.append(
            [np.linalg.norm(delta), np.arctan2(delta[1], delta[0]) - x[2]])

    # Compute the difference between the expected and actual measurements
    diff = np.array(expected_z) - z
    diff[:, 1] = normalize_angle(diff[:, 1])

    # Compute the likelihood of the particle based on the difference
    # We assume that the measurement noise is Gaussian with standard deviation of 0.2 meters and 0.2 radians
    cov = np.diag([0.2 ** 2, (0.2 * np.pi / 180) ** 2])
    likelihood = np.exp(
        -0.5 * np.sum((diff @ np.linalg.inv(cov) * diff), axis=1))
    return likelihood


def sample_motion_model_odometry(particle, odometry):
    """
    Updates the pose of a particle based on the odometry measurement.
    """
    alpha1, alpha2, alpha3, alpha4 = 1, 1, 1, 1

    x, y, theta = particle
    delta_x, delta_y, delta_theta = odometry

    # Sample from Gaussian distribution with mean = measured delta_x and std. dev. = alpha1*delta_x + alpha2*delta_theta
    std_dev_x = alpha1 * np.abs(delta_x) + alpha2 * np.abs(delta_theta)
    delta_x_hat = np.random.normal(loc=delta_x, scale=std_dev_x)

    # Sample from Gaussian distribution with mean = measured delta_y and std. dev. = alpha1*delta_y + alpha2*delta_theta
    std_dev_y = alpha1 * np.abs(delta_y) + alpha2 * np.abs(delta_theta)
    delta_y_hat = np.random.normal(loc=delta_y, scale=std_dev_y)

    # Sample from Gaussian distribution with mean = measured delta_theta and std. dev. = alpha3*delta_theta + alpha4*delta_x
    std_dev_theta = alpha3 * np.abs(delta_theta) + alpha4 * np.abs(delta_x)
    delta_theta_hat = np.random.normal(loc=delta_theta, scale=std_dev_theta)

    # Update pose of particle
    x += delta_x_hat * np.cos(theta + delta_theta_hat / 2)
    y += delta_y_hat * np.sin(theta + delta_theta_hat / 2)
    theta += delta_theta_hat

    return np.array([x, y, theta])


def update_particles(particles, z, map):
    """
    Updates the particles based on the sensor measurement (z) and the map of landmarks.
    """
    weights = np.zeros(len(particles))
    for i in range(len(particles)):
        # Update particle pose using odometry model
        particles[i] = sample_motion_model_odometry(particles[i], odometry)

        # Compute particle weight using sensor model
        weights[i] = sensor_model(particles[i], z, map)

    # Normalize weights
    weights /= np.sum(weights)

    # Resample particles
    indices = np.random.choice(range(len(particles)), size=len(particles),
                               replace=True, p=weights)
    particles = particles[indices]

    return particles


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


def get_landmark_pos(x, z):
    """
    Returns the position of a detected landmark in the world frame, based on the robot's pose (x) and the range-bearing measurement (z).
    If the detected landmark is not in the map, returns None.
    """
    pos = np.array(
        [x[0] + z[0] * np.cos(x[2] + z[1]), x[1] + z[0] * np.sin(x[2] + z[1])])
    if pos[0] < 0 or pos[0] > map_size or pos[1] < 0 or pos[1] > map_size:
        return None
    else:
        return pos


def update_map(map, particles, weights):
    """
    Updates the occupancy grid map based on the particles and their weights.
    The occupancy grid map is a 2D numpy array of size (grid_size, grid_size), where each cell represents a 20cm x 20cm area in the world frame.
    A cell is marked as occupied if its occupancy probability is greater than 0.5, and as free otherwise.
    """
    # Reset the map
    map.fill(0.5)

    # Compute the occupancy probabilities for each cell in the map
    for i in range(particles.shape[0]):
        pos = particles[i][:2]
        cell = np.array(
            [int(pos[0] / map_resolution), int(pos[1] / map_resolution)])
        if cell[0] < grid_size and cell[1] < grid_size:
            map[cell[1], cell[0]] += weights[i]
    map[map >= 0.5] = 1
    map[map < 0.5] = 0
    return map


def normalize_angle(angle):
    """
    Normalizes an angle between -pi and pi.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Run the FastSLAM algorithm
for t in range(len(odom_data)):
    # Motion update
    u = odom_data[t]
    dt = u[2]
    for i in range(N):
        particles[i] = motion_model(particles[i], u, dt)

    # Measurement update
    z = lidar_data[t]
    for i in range(N):
        weights[i] = 1
        for j in range(len(z)):
            landmark_pos = get_landmark_pos(particles[i], z[j])
            if landmark_pos is not None:
                weights[i] *= sensor_model(z[j], landmark_pos, R)
        weights[i] += 1.e-300  # avoid zero weight
    weights /= np.sum(weights)

    # Resampling
    if 1 / np.sum(weights ** 2) < N / 2:
        indexes = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indexes]
        weights = np.ones(N) / N

    # Update the map
    grid_map = update_map(grid_map, particles, weights)

    # Plot the particles and the map
    plt.clf()
    plt.scatter(particles[:, 0], particles[:, 1], marker='.', color='r')
    plt.imshow(grid_map, origin='lower')
    plt.xlim([0, grid_size])
    plt.ylim([0, grid_size])
    plt.pause(0.001)

plt.show()
