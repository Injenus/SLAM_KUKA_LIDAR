import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.spatial import KDTree
from scipy.linalg import orthogonal_procrustes
import numpy as np
from scipy.spatial import KDTree
import cv2
import numpy as np
import sys
from numpy.random import *

matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1
NUM_FRAMES = 1600  # NUM OF DATA (1600 max)
file_name = 'all_data_like_pd.npy'
data = np.load(file_name, allow_pickle=True)
"""
data = [ [x,y,w,[lidar_data_list]
                        ...
                        ]
that is data[i]=[x_i, y_i, w_i,[lidar_data_list_i]

points1 = np.array([[x1, y1], [x2, y2], ..., [xn1, yn1]])
points2 = np.array([[x1, y1], [x2, y2], ..., [xn2, yn2]])
"""

points1 = np.load('pr_a.npy', allow_pickle=True)
points2 = np.load('cr_a.npy', allow_pickle=True)

from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance


def get_obstacle_points(map):
    obstacle_points = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                obstacle_points.append(
                    (j, i))  # note the order of (x,y) coordinates
    return obstacle_points


map1_points = get_obstacle_points(points1)
map2_points = get_obstacle_points(points2)

# Find the centroids of the two point sets
centroid1 = np.mean(map1_points, axis=0)
centroid2 = np.mean(map2_points, axis=0)

# Calculate the translation that brings centroid2 to centroid1
translation = centroid1 - centroid2

# Calculate the rotation that brings the two centroids into alignment
angle = np.arctan2(centroid1[1] - centroid2[1], centroid1[0] - centroid2[0])
rotation = np.array(
    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

# Define the initial transformation matrix
initial_transform = np.hstack((rotation, translation.reshape(-1, 1)))


def icp(source_points, target_points, initial_transform=None, max_iterations=100, tolerance=1e-8):
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    if initial_transform is None:
        initial_transform = np.eye(3)
    else:
        initial_transform = np.vstack((initial_transform, np.array([0, 0, 1])))

    source_tree = KDTree(source_points)

    for i in range(max_iterations):
        # Transform the source points using the current transformation
        transformed_points = np.dot(initial_transform, np.vstack((source_points.T, np.ones(source_points.shape[0]))))
        transformed_points = transformed_points[:2, :].T

        # Find the nearest neighbors between the transformed source points and the target points
        distances, indices = source_tree.query(transformed_points, distance_upper_bound=0.1)

        # Remove any indices that are out of bounds
        indices = indices[~np.isinf(distances)]

        # Construct the corresponding pairs of points
        source_correspondences = source_points[indices]
        target_correspondences = target_points[indices]

        # Calculate the transformation that aligns the corresponding pairs of points
        source_mean = np.mean(source_correspondences, axis=0)
        target_mean = np.mean(target_correspondences, axis=0)

        centered_source = source_correspondences - source_mean
        centered_target = target_correspondences - target_mean

        covariance_matrix = np.dot(centered_source.T, centered_target)
        u, s, vt = np.linalg.svd(covariance_matrix)

        rotation_matrix = np.dot(vt.T, u.T)
        translation_vector = target_mean - np.dot(rotation_matrix, source_mean)

        # Update the transformation using the calculated rotation and translation
        transform = np.eye(3)
        transform[:2, :2] = rotation_matrix
        transform[:2, 2] = translation_vector

        initial_transform = np.dot(initial_transform, transform)

        # Check for convergence
        if np.linalg.norm(transform - np.eye(3)) < tolerance:
            break

    return initial_transform

a=icp(map1_points, map2_points)
