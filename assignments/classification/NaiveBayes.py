from itcs4156.models.ClassificationModel import ClassificationModel
import numpy as np


class NaiveBayes(ClassificationModel):
    def __init__(self, alpha=1.0):
        ClassificationModel.__init__(self)
        self.alpha = alpha

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

    def train(self, X, Y):
        D = X.shape[0]

        N = [[x for x, y in zip(X, Y) if y == i] for i in np.unique(Y)]
        self.class_likeihood_ = [np.log(len(i) / D) for i in N]

        K = np.array([np.array(i).sum(axis=0) for i in N]) + self.alpha
        self.feature_likeihood_ = np.log(K / K.sum(axis=1)[np.newaxis].T)

        return self

    def likeihood(self, X):
        return [
            (self.feature_likeihood_ * x).sum(axis=1) + self.class_likeihood_ for x in X
        ]

    def predict(self, X):
        # make predictions about the class of inputs. for eg random prediction
        ## TODO: replace this with your codes
        return np.argmax(self.likeihood(X), axis=1)
