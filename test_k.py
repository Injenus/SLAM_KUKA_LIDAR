import numpy as np
# Process and sensing noise covariances.
V = np.array([[params["V_00"],0.0],[0.0,params["V_11"]]])
W = np.array([[params["W_00"],0.0],[0.0,params["W_11"]]])