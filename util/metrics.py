import numpy as np

def mean_sq_error(Y, T):
    """
      Y : Ground Truth (Target)
      T : Predictions
    """
    m = Y.shape[1]
    T = T.reshape(-1,m)
    mse = np.mean((Y - T) ** 2)
    return mse