from abc import abstractclassmethod
import numpy as np
from tqdm import tqdm


class SingleLayerNetwork:
    """
    A class for representing a Neural Network with one hidden layer.
    """

    def __init__(
        self,
        num_input_features,
        num_hidden_features,
        num_output_features,
        hidden_activation=None,
        output_activation=None,
    ):
        """
        parameters
        -----------
        num_input_features      int
                                number of units in the input layer
        num_hidden_features     int
                                number of units in the hidden layer
        num_output_features     int
                                number of units in the output layer
        hidden_activation       str
                                choice of activation function to apply to the output of hidden layer.
                                Default: None -> No activation
                                Should accept the following values:
                                    None, 'tanh'
        output_activation       str
                                choice of activation function to apply to the output of final/output layer.
                                Default: None -> No activation
                                Should accept the following values:
                                    None, 'softmax'
        """
        self.num_input_features = num_input_features
        self.num_hidden_features = num_hidden_features
        self.num_output_features = num_output_features
        self.hidden_activation = None
        self.output_activation = None

    def init_weights(self):
        """
        The method to reset/initialize the weights of the neural network.
        """
        # TODO: Initialize weights
        K = (
            self.num_output_features.shape[1]
            if len(self.num_output_features.shape) == 2
            else 1
        )
        N = self.num_input_features + 1
        M = 1 + self.num_hidden_features
        self.V = 0.1 * 2 * (np.random.uniform(size=(N, self.num_hidden_features)) - 0.5)
        self.W = 0.1 * 2 * (np.random.uniform(size=(M, K)) - 0.5)

    def compute_loss(self, Y_pred, Y_target):
        """
        calculate your loss based on the loss function to use.
        parameters
        ----------
        Y_pred      ndarray
                    Network output

        Y_target    ndarray
                    Target output

        returns
        --------

        loss_val    float
                    total loss value
        """
        # TODO: Implement your loss functions here
        loss = Y_target * -np.log(Y_pred)
        loss_val = np.sum(loss)
        return loss_val

    def forward(self, X):
        """
        The forward pass of the neural network
        parameters
        ---------
        X           ndarray
                    Input to the neural network

        returns
        -------
        in_hidden       ndarray
                        Input to the hidden layer

        out_hidden      ndarray
                        Output of the hidden layer

        in_final        ndarray
                        Input to the final layer

        out_final       ndarray
                        The final output of the network
        """

        # TODO: complete his method to implement forward pass and return the following variables

        in_hidden = self.add_ones(X)
        out_hidden = self.hidden_activation(in_hidden @ self.V)
        in_final = self.add_ones(out_hidden)
        out_final = self.output_activation(in_final, self.W)

        return in_hidden, np.argmax(out_hidden, axis=1), in_final, out_final

    def backward(self, in_hidden, out_hidden, in_final, out_final, Y):
        """
        The backward pass of the neural network
        parameters
        ----------
        Y        ndarray
                Target output

        """

        # TODO: Update your weights here
        # self.V = self.V +
        pass

    def train(self, X_tr, Y_tr, epochs, lr_h, lr_o, loss_f, X_val=None, Y_val=None):
        """
        Method that trains the network
        parameters:
        -----------
        X_tr    ndarray
                Input to the network

        Y_tr    ndarray
                Target output

        epochs  int
                The number of iterations to go over the data

        lr_h    float
                Learning rate of the hidden layer

        lr_o    float
                learning rate of the output layer

        loss_f    str
                loss method to use.
                Should accept the following values:
                "MSE", "cross_entropy"

        X_val   ndarray
                optional. Default = None
                Validation input

        Y_val   ndarray
                optional. Default = None
                Validation output

        Returns
        -------

        trace   ndarray
                loss trace from training and/or validation

        """

        ## ========================================================================= ##

        #    This method has been implemented for us, so no change is required here.
        #    Please read through the code to understand the training process.

        #    Once we correctly implement our TODO in the above functions,
        #    this method should execute smoothly.

        ## ========================================================================= ##

        # Storing learning rates
        self.lr_h = lr_h
        self.lr_o = lr_o

        # Storing loss function to use
        self.loss_f = loss_f

        # Initialize weights of the network
        self.init_weights()

        trace = np.zeros((epochs, 1)) if Y_val is None else np.zeros((epochs, 2))

        # start training
        for i in tqdm(range(epochs)):

            # train
            in_hidden, out_hidden, in_final, out_final = self.forward(X_tr)
            train_loss = self.compute_loss(out_final, Y_tr)
            trace[i, 0] = train_loss
            self.backward(in_hidden, out_hidden, in_final, out_final, Y_tr)

            # validate
            if Y_val is not None:
                _, _, _, out_val = self.forward(X_val)
                val_loss = self.compute_loss(out_val, Y_val)
                trace[i, 1] = val_loss

        return trace

    @abstractclassmethod
    def predict(self, Y):
        """
        Abstract method for making prediction based on the output of the network.
        Classes inheriting this class should implement this method.
        """
        pass
