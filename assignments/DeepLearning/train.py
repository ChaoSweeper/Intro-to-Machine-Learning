class Train:
    class KerasModel:

        # TODO: Set Parameters

        dense_layers = [200, 100, 50, 40, 30, 20, 10]
        activation = "relu"
        regularization = "l2"
        batch_norm = True
        weight_init = 'glorot_uniform'
        batch_size = 32
        optimizer = "Adam"
        learning_rate = 0.0001
        epochs = 30