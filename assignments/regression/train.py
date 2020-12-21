from itcs4156.util.data import get_input_output
from itcs4156.assignments.regression.LeastSquares import LeastSquares
from itcs4156.assignments.regression.LeastMeanSquares import LMS
from itcs4156.assignments.regression.PolynomialSimple import PolynomialSimple
from itcs4156.assignments.regression.PolynomialMulti import PolynomialRegression


def train_ls(X, Y):
    model = LeastSquares()
    model.train(X, Y)
    return model


def train_lsm(X, Y):
    alpha = 0.01  # TODO Set your learning rate
    model = LMS(alpha)
    model.train(X, Y)
    return model


def train_poly_simple(X, Y):
    degree = 3  # TODO Set the degree of your polynomial here
    model = PolynomialSimple(degree)
    model.train(X, Y)
    return model


def train_poly_multi(X, Y):
    degree = 2  # TODO Set the degree
    lamb = 0.1  # TODO Set the value of regularization parameter
    model = PolynomialRegression(degree, lamb)
    model.train(X, Y)
    return model


def get_features_for_lsm():
    features = [
        "CRIM",
        "ZN",
    ]  # TODO fill this list with features you want to use for training LSM
    return features


def get_features_for_poly_multi():
    features = [
        "CRIM",
        "ZN",
        "DIS",
        "RM",
        "RAD",
        "INDUS",
        "AGE",
        "TAX",
        "LSTAT",
    ]  # TODO fill this list with features you want to use for training PolynomialRegression
    return features