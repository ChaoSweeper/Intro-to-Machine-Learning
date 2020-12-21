import numpy as np
import pandas as pd

from itcs4156.assignments.NeuralNetworks.SingleLayerNetwork import SingleLayerNetwork
    

class NNClassifier(SingleLayerNetwork):

    def __init__(self, n_i_f, n_h_f, n_o_f, h_a, o_a):
        SingleLayerNetwork.__init__(self, n_i_f, n_h_f, n_o_f, h_a, o_a)

    def fit_transform(self, X):
        """
            Place to pre-process your training data. This is called before training your model.
            For eg: You can normalize your data or choose to do some feature selection/transformation.
            In the simplest case you can return your data as it is.
        """
        
        # TODO: Pre-processing if required.
        Xt = X
        return Xt

    def transform(self, X):
        """
            This function is called before making a prediction using your model.
            Any statistics/transformation that you learned from your training data in the fit_transform method could be used here.
            In the simplest case, you can return the input as it is.
        """
        
        # TODO: any processing of input before being passed to classify.
        Xt = X
        return Xt

    def train(self, X_tr, Y_tr, 
                    epochs, lr_h, lr_o, loss_f,
                    X_val = None, Y_val = None):

        # Apply transformations
        Xt_tr = self.fit_transform(X_tr)
        if X_val is not None:
            Xt_val = self.transform(X_val) 
        else:
            Xt_val = None
        
        # TODO: (If required) Update any values here before passing them to super().train below
       
        trace = super().train(Xt_tr, Y_tr, 
                        epochs, lr_h, lr_o, loss_f,
                        X_val=Xt_val, Y_val=Y_val)
        
        return trace


    def predict(self, X):
        Xt = self.transform(X)
        
        # TODO: Use the output of your neural network's forward method to make a prediction
        
        # Returns random predictions. Remove this when you implement this method.
        Y = np.random.choice(range(10), Xt.shape[0]) 
        
        return Y