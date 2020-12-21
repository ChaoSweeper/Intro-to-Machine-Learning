# Python Imports
import numpy as np
import random
import traceback
from sklearn.metrics import accuracy_score

# Internal Imports
from itcs4156.util.timer import Timer
from itcs4156.util.metrics import mean_sq_error
from itcs4156.util.data import get_input_output
from itcs4156.datasets.HousingDataset import HousingDataset
from itcs4156.datasets.MNISTDataset import MNISTDataset


SUCCESS = 0
FAILED = -1
NN_REGRESSION = 1
NN_CLASSIFICATION = 2


def score_regression(mse, max_score):
    thresh = 24.0
    if mse <= thresh:
        score_percent = 100
    elif mse is not None:
        score_percent = (thresh / mse) * 100
        if score_percent < 40:
            score_percent = 40
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0
    return score

def score_classification(acc, max_score):
    score_percent = 0
    if acc >= 0.90:
        score_percent = 100
    elif acc >= 0.80:
        score_percent = 90
    elif acc >= 0.70:
        score_percent = 80
    elif acc >= 0.60:
        score_percent = 70
    elif acc >= 0.50:
        score_percent = 60
    elif acc >= 0.40:
        score_percent = 55
    elif acc >= 0.30:
        score_percent = 50
    elif acc >= 0.20:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def build_train_evaluate(MODEL, X_tr, Y_tr, X_val, Y_val):
    try:
        from itcs4156.assignments.NeuralNetworks.train import Train

        if MODEL == NN_REGRESSION:
            print("==== Testing NNRegression ====\n")
            from itcs4156.assignments.NeuralNetworks.NNRegressor import NNRegressor
            NNModel = NNRegressor
            Parameter = Train.NNRegressor
            measure = mean_sq_error
            name = "MSE"
        
        elif MODEL == NN_CLASSIFICATION:
            print("\n==== Testing NNClassification ====\n")
            from itcs4156.assignments.NeuralNetworks.NNClassifier import NNClassifier
            NNModel = NNClassifier
            Parameter = Train.NNClassifier
            measure = accuracy_score
            name = "Accuracy"
        
        else:
            raise ValueError("Unknown model: ", MODEL)

        timer  = Timer()
        timer.start()
        
        print("1. Building NN with Parameters: \n ", 
            "Input Units: {}, Hidden Units: {}, Output Units: {}, Hidden Activation: {}, Output Activation: {}".format( 
                Parameter.num_input, Parameter.num_hidden, Parameter.num_output, 
                Parameter.hidden_activation, Parameter.output_activation))
        
        model = NNModel(n_i_f = Parameter.num_input,
                        n_h_f = Parameter.num_hidden,
                        n_o_f = Parameter.num_output,
                        h_a = Parameter.hidden_activation,
                        o_a = Parameter.output_activation)

        print("2. Training NN with Parameters: \n ",
            "Epochs: {}, lr_h: {}, lr_o: {}, loss_f: {}".format(
                Parameter.epochs, Parameter.lr_h, Parameter.lr_o, Parameter.loss_f))

        trace = model.train(X_tr, Y_tr, Parameter.epochs, 
            Parameter.lr_h, Parameter.lr_o, Parameter.loss_f)
        
        print("3. Evaluating Training Performance: ")
        Y_tr_pred = model.predict(X_tr)
        metric_tr = measure(Y_tr, Y_tr_pred)
        print(" {} = {}".format(name, metric_tr))

        print("4. Evaluating Validation Performance: ")
        Y_val_pred = model.predict(X_val)
        metric_val = measure(Y_val, Y_val_pred)
        print(" {} = {}\n".format(name, metric_val))

        timer.stop()

        exit_status = SUCCESS
    
    except Exception as e:
        track = traceback.format_exc()
        print("The following error occured while trying to evaluate the model: \n", track)
        exit_status = FAILED
        metric_tr = None
        metric_val = None

    return exit_status, metric_tr, metric_val


def eval():
    main_timer = Timer()
    main_timer.start()

    seed = 25
    print("Setting Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)

    # NNRegression
    dataset = HousingDataset()
    df_train, df_val = dataset.load()
    in_feature = list(df_train.columns[:-1])
    X_tr, Y_tr = get_input_output(df_train, in_feature, "MEDV")
    X_val, Y_val = get_input_output(df_val, in_feature, "MEDV")
    status, mse_tr, mse_val = build_train_evaluate(NN_REGRESSION, X_tr, Y_tr, X_val, Y_val)
    score_a = score_regression(mse_val, 40)
    print("Score: {:2f}\n".format(score_a))

    # NNClassification
    dataset = MNISTDataset()
    train_data, val_data = dataset.load()
    X_tr, Y_tr = train_data
    X_val, Y_val = val_data
    status, acc_tr, acc_val = build_train_evaluate(NN_CLASSIFICATION, X_tr, Y_tr, X_val, Y_val)
    score_b = score_classification(acc_val, 40)
    print("Score: {:2f}\n".format(score_b))

    total = round(score_a + score_b)
    print("Totals\nScore: {}/80".format(total))
    main_timer.stop()

if __name__ == "__main__":
    eval()

