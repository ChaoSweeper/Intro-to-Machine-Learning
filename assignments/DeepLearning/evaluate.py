# Python Imports
import numpy as np
import random
import traceback
from sklearn.metrics import accuracy_score


# Internal Imports
from itcs4156.util.timer import Timer
from itcs4156.datasets.FMNISTDataset import FMNISTDataset

def score_classification(acc, max_score):
    if acc >= 0.9:
        score_percent = 100
    else:
        score_percent = (acc / 0.9) * 100 
        if score_percent < 40:
            score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def eval():

    main_timer = Timer()
    main_timer.start()

    seed = 25
    print("Setting Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = FMNISTDataset()
    (X_tr, Y_tr), (X_val, Y_val) = dataset.load()

    try:
        from itcs4156.assignments.DeepLearning.KerasModel import KerasModel
        from itcs4156.assignments.DeepLearning.train import Train

        print("\n1. Initializing model with parameters: \n",
            """dense_layers: {}, activation: {}, regularization: {}, 
            batch_norm: {}, weight_init: {}, batch_size: {},
            optimizer: {}, learning_rate: {}, epochs: {}\n""".format(
                   Train.KerasModel.dense_layers, Train.KerasModel.activation,
                   Train.KerasModel.regularization, Train.KerasModel.batch_norm, 
                   Train.KerasModel.weight_init,Train.KerasModel.batch_size, 
                   Train.KerasModel.optimizer, Train.KerasModel.learning_rate,
                   Train.KerasModel.epochs))
        
        model = KerasModel(Train.KerasModel.dense_layers, Train.KerasModel.activation,
                   Train.KerasModel.regularization, Train.KerasModel.batch_norm, 
                   Train.KerasModel.weight_init,Train.KerasModel.batch_size, 
                   Train.KerasModel.optimizer, Train.KerasModel.learning_rate,
                   Train.KerasModel.epochs, "Test")
        
        print("\n2. Building model:")
        sequential = model.build()
        if sequential is not None:
            sequential.summary()

        print("\n3. Training: ")
        history = model.train(X_tr, Y_tr, X_val, Y_val)
        
        print("\n4. Evaluating: ")
        Y_pred = model.predict(X_val)
        acc = accuracy_score(Y_val, Y_pred)


    except Exception as e:
        track = traceback.format_exc()
        acc = 0
        print("The following error occured while trying to evaluate the model: \n", track)
    
    print("\nAccuracy: ", acc)
    score = round(score_classification(acc, 50))
    print("Score: {}/50".format(score))
    main_timer.stop()

if __name__ == "__main__":
    eval()

