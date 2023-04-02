import numpy as np
import matplotlib.pyplot as plt
prob_map = np.load('occupancy_grid.npy')
print(prob_map.shape)
plt.imshow(prob_map)
plt.show()
