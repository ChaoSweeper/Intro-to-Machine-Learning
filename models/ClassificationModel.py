import numpy as np
from abc import abstractclassmethod
from itcs4156.models.BaseModel import BaseModel


class ClassificationModel(BaseModel):
    """
    Abstract class for classification

    Attributes
    ==========
    """

    @abstractclassmethod
    def fit_transform(self, X):
        """
        Place to pre-process your training data. This is called before training your model.
        For eg: You can normalize your data or choose to do some feature selection/transformation.
        In the simplest case you can return your data as it is.
        """
        # TODO: Pre-processing if required.
        pass

    @abstractclassmethod
    def transform(self, X):
        """
        This function is called before making a classification using your model.
        Any statistics/transformation that you learned from your training data in the fit_transform method could be used here.
        In the simplest case, you can return the input as it is.
        """

        # TODO: processing of input before being passed to predict.

        pass

    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError("".join(["Wrong matrix ", name]))

    # add a basis
    def add_ones(self, X):
        """
        add a column basis to X input matrix
        """
        self._check_matrix(X, "X")
        return np.hstack((np.ones((X.shape[0], 1)), X))

    ####################################################
    #### abstract funcitons ############################
    @abstractclassmethod
    def train(self, X, Y):
        pass

    @abstractclassmethod
    def predict(self, Y):
        pass
