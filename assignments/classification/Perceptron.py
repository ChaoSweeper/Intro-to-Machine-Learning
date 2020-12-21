from itcs4156.models.ClassificationModel import ClassificationModel
import numpy as np


class Perceptron(ClassificationModel):
    def __init__(self, alpha, epochs):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.bias = None

    def fit_transform(self, X):
        """
        Place to pre-process your training data. This is called before training your model.
        For eg: You can normalize your data or choose to do some feature selection/transformation.
        In the simplest case you can return your data as it is.
        """
        return X

    def transform(self, X):
        """
        This function is called before making a classification/prediction using your model.
        Any statistics/transformation that you learned from your training data in the fit_transform method could be used here.
        In the simplest case, you can return the input as it is.
        """
        return X

    def steps(self, x):
        return np.where(x >= 0, 1, 0)

    def train(self, X, Y):
        D, N = X.shape
        self.w = np.zeros(N)
        self.bias = 0

        ys = np.array([1 if i > 0 else 0 for i in Y])

        for _ in range(self.epochs):
            for i, k in enumerate(X):
                output = np.dot(k, self.w) + self.bias
                yp = self.steps(output)
                update_w = self.alpha * (ys[i] - yp)
                self.w += update_w * k
                self.bias += update_w

    def predict(self, X):
        # make predictions about the class of inputs. for eg random prediction
        ## TODO: replace this with your codes
        output = np.dot(X, self.w) + self.bias
        Y = self.steps(output)
        return Y
