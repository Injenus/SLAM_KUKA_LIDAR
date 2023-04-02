import numpy as np
# Define the state vector
x = np.array([[0, 0, 0],[1,2,3],[4,5,6]])
print('t',x[0:3, 0:2])
print('j',x[:,2])

# Define the error covariance matrix
P = np.diag([0.1, 0.1, 0.1])

print(P)

grid_origin = np.array([-25.0, -25.0])
print(grid_origin)