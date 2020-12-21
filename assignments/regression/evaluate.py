import os
from itcs4156.util.data import get_input_output, mean_normalize
from itcs4156.datasets.HousingDataset import HousingDataset
from itcs4156.util.metrics import mean_sq_error
from itcs4156.assignments.regression.train import train_ls, train_lsm, train_poly_multi, train_poly_simple
from itcs4156.assignments.regression.train import get_features_for_lsm, get_features_for_poly_multi

import numpy as np
import random


def model_mse(model, X_tr, Y_tr, X_val, Y_val):
    P_tr = model.predict(X_tr)
    P_val = model.predict(X_val)
    mse_tr = mean_sq_error(Y_tr, P_tr)
    mse_val = mean_sq_error(Y_val, P_val)
    return mse_tr, mse_val

def eval():
    np.random.seed(25)
    random.seed(25)

    LMS_features = get_features_for_lsm()
    Poly_features = get_features_for_poly_multi()
    
    if "MEDV" in LMS_features or "MEDV" in Poly_features:
        print("The target var MEDV can not be used as feature!")
        return
    
    model_features = ["RM", LMS_features, "LSTAT", Poly_features]
    train_models = [train_ls, train_lsm, train_poly_simple, train_poly_multi]
    model_names = ["LeastSquares", "LMS", "PolynomialSimple", "PolynomialRegression"]
    model_threshold_mse = [0.60, 0.90, 0.30, 0.30]
    dataset = HousingDataset()
    df_train, df_val = dataset.load()
    df_train_norm, mu, sigma = mean_normalize(df_train)
    df_val_norm, _, _ = mean_normalize(df_val, mu, sigma)
    avg_train_mse = 0.0
    avg_val_mse = 0.0
    test_passed = 0

    for model_name, in_feature, train_model, th in zip(model_names, model_features, train_models, model_threshold_mse):

        X_tr, Y_tr = get_input_output(df_train_norm, in_feature, "MEDV")
        X_val, Y_val = get_input_output(df_val_norm, in_feature, "MEDV")

        print("Training: ", model_name)
        model = train_model(X_tr, Y_tr)
        
        print("Evaluating: ", model_name)
        train_mse, val_mse = model_mse(model, X_tr, Y_tr, X_val, Y_val)
        print("MSE: Train = {}, Val = {}".format(train_mse, val_mse))
        status = val_mse < th
        print("Is Val MSE < {}: {}".format(th, status))
        print("Test {}\n".format("PASSED! (っ＾▿＾)💨" if status else "FAILED! (ノಠ益ಠ)ノ彡┻━┻"))

        test_passed += status

       
        avg_train_mse += train_mse
        avg_val_mse += val_mse

    avg_train_mse = avg_train_mse / 4.0
    avg_val_mse = avg_val_mse / 4.0

    score = test_passed * 20
    print("Tests Passed: {}/4, Score: {}/80\n".format(test_passed, score))
    print("Computing Averages")
    print("Average Training MSE: ", avg_train_mse)
    print("Average Validation MSE: ", avg_val_mse)
    

if __name__ == "__main__":
    eval()
  




    


