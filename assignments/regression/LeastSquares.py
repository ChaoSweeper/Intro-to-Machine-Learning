from itcs4156.models.LinearModel import LinearModel
import numpy as np


# Linear Regression Class for least squares
class LeastSquares(LinearModel):
    """
    LeastSquares class

    attributes
    ===========
    w    nd.array  (column vector/matrix)
    weights
    """

    def __init__(self):
        LinearModel.__init__(self)

    # train least-squares model
    def train(self, X, Y):
        X = self.add_ones(X)
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ Y

    # apply the learned model to data X
    def predict(self, X):
        X = self.add_ones(X)
        return X @ self.w
