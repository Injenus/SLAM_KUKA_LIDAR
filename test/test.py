import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the SLAM algorithm using EKF-SLAM
def slam(odometry_data, lidar_data, Q, R):
    # Define the state vector
    state = np.zeros(3 + 2 * len(lidar_data))
    state[0] = odometry_data[0]
    state[1] = odometry_data[1]
    state[2] = odometry_data[2]

    # Define the covariance matrix
    P = np.zeros((3 + 2 * len(lidar_data), 3 + 2 * len(lidar_data)))

    # Define the motion and measurement models
    def motion_model(state, u):
        x = state[0]
        y = state[1]
        theta = state[2]
        delta_x = u[0]
        delta_y = u[1]
        delta_theta = u[2]

        x += delta_x * np.cos(theta) - delta_y * np.sin(theta)
        y += delta_x * np.sin(theta) + delta_y * np.cos(theta)
        theta += delta_theta

        return np.array([x, y, theta])

    def measurement_model(state, landmark_idx):
        x = state[0]
        y = state[1]
        theta = state[2]
        l_x = state[3 + 2 * landmark_idx]
        l_y = state[3 + 2 * landmark_idx + 1]

        delta_x = l_x - x
        delta_y = l_y - y

        q = delta_x ** 2 + delta_y ** 2
        z = np.array([
            np.sqrt(q),
            np.arctan2(delta_y, delta_x) - theta
        ])

        return z, np.array([
            [delta_x / np.sqrt(q), delta_y / np.sqrt(q), 0],
            [-delta_y / q, delta_x / q, -1]
        ])

    # Define the Jacobian of the motion model
    def jacobian_motion_model(state, u):
        theta = state[2]
        delta_x = u[0]
        delta_y = u[1]

        return np.array([
            [1, 0, -delta_x * np.sin(theta) - delta_y * np.cos(theta)],
            [0, 1, delta_x * np.cos(theta) - delta_y * np.sin(theta)],
            [0, 0, 1]
        ])

    # Define the Jacobian of the measurement model
    def jacobian_measurement_model(state, landmark_idx):
        x = state[0]
        y = state[1]
        theta = state[2]
        l_x = state[3 + 2 * landmark_idx]
        l_y = state[3 + 2 * landmark_idx + 1]

        delta_x = l_x - x
        delta_y = l_y - y
        q = delta_x ** 2 + delta_y ** 2

        return np.array([
            [-delta_x / np.sqrt(q), -delta_y / np.sqrt(q), 0, delta_x / np.sqrt(q), delta_y / np.sqrt(q)],
            [delta_y / q, -delta_x / q, -1, -delta_y / q, delta_x / q]
        ])

    # Loop over the lidar measurements
    for i, z in enumerate(lidar_data):
        # Perform prediction step
        delta_x



# Read in the odometry and lidar data
odometry_data = np.loadtxt('odometry_data.txt')
lidar_data = np.loadtxt('lidar_data.txt')

# Run the SLAM algorithm and get the position and karta
position, map = slam(odometry_data, lidar_data)


# Define the animation function
def animate(i):
    # Update the position and karta based on the new odometry and lidar data
    position, map = slam(odometry_data[i], lidar_data[i])

    # Plot the karta and the robot's position
    plt.clf()
    plt.plot(map[:, 0], map[:, 1], 'ko')
    plt.plot(position[0], position[1], 'ro')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title('SLAM Animation')
    plt.xlabel('X')
    plt.ylabel('Y')


# Create the animation
ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(odometry_data),
                              interval=50)

# Show the animation
plt.show()
