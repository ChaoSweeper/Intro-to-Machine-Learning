import tensorflow as tf
from tensorflow import keras
import numpy as np


class KerasModel:
    """
    Class to instantiate and train a tf.keras.Model using the given parameters.
    """

    def __init__(
        self,
        dense_layers=[20, 10],
        activation="relu",
        regularization=None,
        batch_norm=False,
        weight_init="glorot_uniform",
        batch_size=32,
        optimizer="Adam",
        learning_rate=0.001,
        epochs=20,
        name="default",
    ):
        """
        parameters
        ----------
        dense_layers    List[int]
                        size of dense layers in the network.
                        default: [20,10]
                        for eg: [10,5,1] represents a network with
                        3 dense layers with number of units 10, 5 and 1 respectively.


        activation      str
                        activation function to use in each dense layer
                        default: 'relu'
                        Should  accept any the following values:
                            'linear', 'relu', 'sigmoid', 'tanh',
                            'softplus', 'elu', 'selu', 'swish'

        regularization  str
                        type of regularization to use in each dense layer
                        default: None
                        Should accept any of the following values:
                            None, L1, L2, L1_L2, dropout

        batch_norm      boolean
                        Whether to use Batch Normalization layer or not
                        default: False

        weight_init     str
                        type of weight initialization to use in each dense layer
                        default: 'glorot_uniform'
                        Should accept any the following values:
                            'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
                            'random_normal', 'random_uniform', 'ones',

        batch_size      int
                        training batch size
                        default: 32

        optimizer       str
                        optimizer to use for training the network
                        default: 'Adam'
                        Should accept any of the following values:
                            'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD'

        learning_rate   float
                        learning rate used by the optimizer
                        default: 0.001

        epochs          int
                        number of training iterations to go over the data
                        default: 20

        name            str
                        A custome name for this model
                        default: "default"
        """

        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size

        # TODO: Initialize rest of the variables
        self.dense_layers = dense_layers
        self.activation = activation
        self.regularization = regularization
        self.batch_norm = batch_norm
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def construct_dense_layer(self, num_units):
        """
        parameters
        ----------
        num_units   int
                    number of units in the dense layer

        returns
        --------
                    An instance of tf.keras.layers.Dense
        """

        # TODO: Based on the given number of hidden units, activation function,
        # regularization and weight initialization method -- initialize and
        # return a dense layer
        if self.regularization == "dropout":
            dense = tf.keras.layers.Dense(
                num_units,
                activation=self.activation,
                kernel_regularizer=None,
                kernel_initializer=self.weight_init,
            )
        else:
            dense = tf.keras.layers.Dense(
                num_units,
                activation=self.activation,
                kernel_regularizer=self.regularization,
                kernel_initializer=self.weight_init,
            )

        return dense

    def build(self):
        """
        returns an instance of tf.keras.Sequential
        """

        # TODO: Write logic to construct and return a keras.Sequential model based
        # on the given parameters in __init__ method.
        #  ->> Make use of construct_dense_layer() method to add dense layers to your model
        #  ->> Add dropout layer and/or batch normalization layer if required after each dense layer, except the last layer.
        #  ->> For the last layer, we don't need to use any activation function or regularization.

        # Step 1: add layers to the following list
        layers = []
        layers.append(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
        for i in range(len(self.dense_layers)):
            layers.append(self.construct_dense_layer(self.dense_layers[i]))
            if self.batch_norm == True:
                layers.append(tf.keras.layers.BatchNormalization())
        layers.append(self.construct_dense_layer(self.dense_layers[-1]))

        # Step 2: construct a sequential model with 'layers' and set the name parameter with self.name
        model = tf.keras.Sequential(name=self.name, layers=layers)

        return model

    def get_optimizer(self):
        """
        returns an instance of tf.keras.Optimizer
        """

        # TODO: based on the given optimizer and learning rate,
        # Initialize and return a keras optimizer
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

        optim = eval(f"tf.keras.optimizers.{self.optimizer}(lr={self.learning_rate})")

        return optim

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

        # TODO: any processing of input before being used for prediction .
        Xt = X

        return Xt

    def train(self, X_tr, Y_tr, X_val=None, Y_val=None, verbose=2):
        """
        parameters
        ----------

        X_tr   ndarray
                Input to the network

        Y_tr    ndarray
                Target output

        X_val   ndarray
                optional. Default = None
                Validation input

        Y_val   ndarray
                optional. Default = None
                Validation output

        verbose 0,2
                0 = silent (no output)
                2 = interactive (outputs for each epoch)

        returns
        -------
        history.history     dict
                            dictionary with fields 'loss', 'accuracy' and
                            additionally with 'val_loss', 'val_accuracy'

        """

        ## ========================================================================= ##

        #    This method has been implemented for you.
        #    Please read through the code to understand the training process.

        #    Once we correctly implement our TODO in the above functions,
        #    this method should execute smoothly.

        #    Don't forget the one TODO in this method.

        ## ========================================================================= ##

        # TODO: COMMENT THE LINE BELOW TO ALLOW EXECUTING THIS METHOD.
        # return {'loss' : [], 'accuracy': [], 'val_loss': [], 'val_accuracy' : []}

        print("Training Model: ", self.name)

        model = self.build()

        model.compile(
            optimizer=self.get_optimizer(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        Xt_tr = self.fit_transform(X_tr)
        if X_val is not None:
            Xt_val = self.transform(X_val)
            validation_data = (Xt_val, Y_val)
        else:
            validation_data = None

        history = model.fit(
            Xt_tr,
            Y_tr,
            validation_data=validation_data,
            epochs=self.epochs,
            verbose=verbose,
            batch_size=self.batch_size,
        )

        # Storing a reference to the trained model
        self.trained_model = model

        return history.history

    def predict(self, X_te):

        Xt_te = self.transform(X_te)

        # TODO: Use your trained model to make predictions
        Y = self.trained_model.predict(Xt_te)

        # Returns random predictions. Remove this when you implement this method.
        # Y = np.random.choice(range(10), Xt_te.shape[0])
        return np.argmax(Y, 1)

    def evaluate(self, X_te, Y_te):

        print("Evaluating Model: ", self.name)

        Xt_te = self.transform(X_te)

        # TODO: Evaluate your trained model on the given data to return loss and accuracy
        error = self.trained_model.evaluate(Xt_te, Y_te)
        # Returns temporary values. Remove the code below when you are done with TODO.
        # loss = error[0]
        # acc = error[1]

        return error
