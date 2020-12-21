from itcs4156.assignments.regression.LeastSquares import LeastSquares
import numpy as np


class PolynomialSimple(LeastSquares):
    """
    PolynomialSimple class

    attributes
    ==========
    w   nd.array  (column vector/matrix)
        weights
    """

    def __init__(self, degree):
        LeastSquares.__init__(self)
        self.degree = degree

    def transform(self, X):

        # Implement this method to return polynomial features for the given degree.
        X_poly = []
        for i in range(self.degree + 1):
            X_poly.append(X ** i)
        X_poly = np.hstack(X_poly)
        return X_poly

    # Leave this method as it is.
    def train(self, X, Y):
        X_poly = self.transform(X)
        super().train(X_poly, Y)

    # Leave this method as it is.
    def predict(self, X):
        X_poly = self.transform(X)
        return super().predict(X_poly)
