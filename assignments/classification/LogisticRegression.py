from itcs4156.models.ClassificationModel import ClassificationModel
import numpy as np


class LogisticRegression(ClassificationModel):
    def __init__(self, alpha, epochs):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.w = None

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

    def normalize(self, X):
        means, stds = np.mean(X, 0), np.std(X, 0)
        X = (X - means) / stds
        return X

    def softmax(self, z):
        z_norm = z - np.max(z, axis=1).reshape(-1, 1)
        f = np.exp(z_norm)
        return f / (np.sum(f, axis=1, keepdims=True))

    def g(self, X, w):
        return self.softmax(X @ w)

    def train(self, X, Y):
        Xt = self.fit_transform(X)
        Yt = (Y == np.unique(Y)).astype(int)

        D = Xt.shape[1]
        K = len(np.unique(Y))

        X1 = self.add_ones(Xt)

        self.w = np.random.rand(D + 1, K)

        for _ in range(self.epochs):
            ys = self.g(X1, self.w)
            self.w += self.alpha * X1.T @ (Yt - ys)

    def predict(self, X):
        Xt = self.transform(X)
        # make predictions about the class of inputs. for eg random prediction
        ## TODO: replace this with your code
        X1 = self.add_ones(Xt)
        Y = self.g(X1, self.w)
        return np.argmax(Y, 1)
