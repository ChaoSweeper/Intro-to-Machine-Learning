from itcs4156.assignments.regression.LeastSquares import LeastSquares
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(LeastSquares):
    """
    PolynomialRegression class

    attributes
    ==========
    w   nd.array  (column vector/matrix)
        weights
    """

    def __init__(self, degree, lamb):
        LeastSquares.__init__(self)
        self.degree = degree
        self.lamb = lamb

    def transform(self, X):
        # Implement this method to return polynomial features for the given degree.
        # Protip: Look up "sklearn.preprocessing.PolynomialFeatures"
        X_poly = PolynomialFeatures(self.degree).fit_transform(X)
        return X_poly

    def train(self, X, Y):
        X_poly = self.transform(X)
        X_poly = np.c_[np.ones((X.shape[0])), X_poly]
        if Y is not None:
            self.w = np.linalg.lstsq(
                X_poly.T @ X_poly + self.lamb * np.eye(X_poly.shape[1]),
                X_poly.T @ Y,
                rcond=1,
            )[0]
            return X_poly, self.w

    def predict(self, X):
        X_poly = self.transform(X)
        return super().predict(X_poly)
