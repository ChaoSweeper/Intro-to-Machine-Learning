from itcs4156.datasets.MNISTDataset import MNISTDataset
from itcs4156.assignments.classification.LogisticRegression import LogisticRegression
from itcs4156.assignments.classification.Perceptron import Perceptron
from itcs4156.assignments.classification.NaiveBayes import NaiveBayes
from itcs4156.assignments.classification.train import Train
from itcs4156.util.data import copy_dataset
from itcs4156.util.data import filter_data
from sklearn.metrics import accuracy_score
from itcs4156.util.timer import Timer
import numpy as np
import random
import traceback

SUCCESS = 0
FAILED = -1
PERCEPTRON = 1
NAIVE_BAYES = 2
LOGISTIC_REGRESSION = 3

def build_model(n):
    model = None
    try:
        if n == PERCEPTRON:
            print("Initializing Perceptron with alpha = {} and epochs = {}".format(Train.Perceptron_ALPHA, Train.Perceptron_EPOCHS))
            model = Perceptron(Train.Perceptron_ALPHA, Train.Perceptron_EPOCHS)
        elif n == NAIVE_BAYES:
            print("Initializing NaiveBayes")
            model = NaiveBayes()
        elif n == LOGISTIC_REGRESSION:
            print("Initializing LogisticRegression with alpha = {} and epochs = {}".format(Train.LogisticRegression_ALPHA, Train.LogisticRegression_EPOCHS))
            model = LogisticRegression(Train.LogisticRegression_ALPHA, Train.LogisticRegression_EPOCHS)
        else:
            raise ValueError("Unknown model identifier: ", n)
        
        exit_status = SUCCESS

    except Exception as e:
        track = traceback.format_exc()
        print("The following error occured while trying to build the model: \n", track)
        exit_status = FAILED

    return exit_status, model

def train(model, X, Y):
    print("Training: ", model.__class__.__name__)
    
    try:
        model.train(X, Y)
        exit_status = SUCCESS
    
    except Exception as e:
        track = traceback.format_exc()
        print("Exception caught while training {}:\n {}".format(model.__class__.__name__, track))
        exit_status = FAILED
    
    return exit_status

def evaluate(model, X, Y):
    print("Evaluating: ", model.__class__.__name__)
    try:
        P = model.predict(X)
        acc = accuracy_score(Y, P)
        exit_status = SUCCESS

    except Exception as e:
        track = traceback.format_exc()
        print("Exception caught while evaluating {}:\n {}".format(model.__class__.__name__, track))
        acc = 0
        exit_status = FAILED
    
    return exit_status, acc

def train_and_evaluate(model, X_tr, Y_tr, X_val, Y_val):
    timer  = Timer()
    timer.start()
    status = train(model, X_tr, Y_tr)
    train_acc = 0
    val_acc = 0
    
    if status == SUCCESS:
        status, train_acc = evaluate(model, X_tr, Y_tr)
    
    if status == SUCCESS:
        status, val_acc = evaluate(model, X_val, Y_val)
    
    timer.stop()
    print("Accuracy: Train = {}, Val = {}".format(train_acc, val_acc))
    return train_acc, val_acc

def score_perceptron(acc, max_score):
    score_percent = 0
    if acc >= 0.8:
        score_percent = 100
    elif acc >= 0.75:
        score_percent = 90
    elif acc >= 0.70:
        score_percent = 80
    elif acc >= 0.65:
        score_percent = 70
    elif acc >= 0.60:
        score_percent = 60
    elif acc >= 0.55:
        score_percent = 50
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score
    
def score(acc, max_score):
    score_percent = 0
    if acc >= 0.80:
        score_percent = 100
    elif acc >= 0.70:
        score_percent = 90
    elif acc >= 0.60:
        score_percent = 80
    elif acc >= 0.50:
        score_percent = 70
    elif acc >= 0.40:
        score_percent = 60
    elif acc >= 0.30:
        score_percent = 55
    elif acc >= 0.20:
        score_percent = 50
    elif acc >= 0.15:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def eval():
    main_timer = Timer()
    main_timer.start()

    np.random.seed(25)
    random.seed(25)

    dataset = MNISTDataset()
    train_data, val_data = dataset.load()
    X_tr, Y_tr = train_data
    X_val, Y_val = val_data

    # Perceptron
    XP_tr, YP_tr, XP_val, YP_val = copy_dataset(X_tr, Y_tr, X_val, Y_val)
    XP_tr, YP_tr = filter_data(XP_tr, YP_tr, classes=[0,1])
    XP_val, YP_val = filter_data(XP_val, YP_val, classes=[0,1])
    P_train_acc = 0; P_val_acc = 0
    status, model = build_model(PERCEPTRON)
    if status == SUCCESS:
        P_train_acc, P_val_acc = train_and_evaluate(model, XP_tr, YP_tr, XP_val, YP_val)
    P_score = score_perceptron(P_val_acc, max_score=20)
    print("Score: {:2f}\n".format(P_score))

   
    # Naive Bayes
    XN_tr, YN_tr, XN_val, YN_val = copy_dataset(X_tr, Y_tr, X_val, Y_val)
    N_train_acc = 0; N_val_acc = 0
    status, model = build_model(NAIVE_BAYES)
    if status == SUCCESS:
        N_train_acc, N_val_acc = train_and_evaluate(model, XN_tr, YN_tr, XN_val, YN_val)
    N_score = score(N_val_acc, max_score=20)
    print("Score: {:2f}\n".format(N_score))

    # Logistic Regression
    XL_tr, YL_tr, XL_val, YL_val = copy_dataset(X_tr, Y_tr, X_val, Y_val)
    L_train_acc = 0; L_val_acc = 0
    status, model = build_model(LOGISTIC_REGRESSION)
    if status == SUCCESS:
        L_train_acc, L_val_acc = train_and_evaluate(model, XL_tr, YL_tr, XL_val, YL_val)
    L_score = score(L_val_acc, max_score=40)
    print("Score: {:2f}\n".format(L_score))

    total = round(P_score + N_score + L_score)
    print("Totals\nScore: {}/80".format(total))
    main_timer.stop()

if __name__ == "__main__":
    eval()

