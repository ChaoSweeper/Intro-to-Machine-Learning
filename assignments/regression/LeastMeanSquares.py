from itcs4156.models.LinearModel import LinearModel
import numpy as np


# LMS class
class LMS(LinearModel):
    """
    Lease Mean Squares. online learning algorithm

    attributes
    ==========
    w        nd.array
            weight matrix
    alpha    float
            learning rate
    """

    def __init__(self, alpha):
        LinearModel.__init__(self)
        self.alpha = alpha

    # batch training by using train_step function
    def train(self, X, Y):
        for K in range(X.shape[0]):
            self.train_step(X[K], Y[K])

    # train LMS model one step
    # here the x is 1d vector
    def train_step(self, x, y):
        if self.w is None:
            self.w = np.ones(len(x) + 1)
        x = np.insert(x, 0, 1)
        self.w -= self.alpha * (np.outer((self.w.T @ x) - y, x).reshape(self.w.shape))

    # apply the current model to data X
    def predict(self, X):
        X = self.add_ones(X)
        self.w = self.w.reshape(len(self.w), 1)
        return X @ self.w