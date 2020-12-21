import pandas as pd
from copy import deepcopy as copy

def get_input_output(df, in_features, out_feature):
    """
        constructs (input, output) pair from pandas dataframe

        df: dataframe
        in_features: str or list of str 
        out_feature: target field
        
    """

    if isinstance(in_features, list):
        n = len(in_features)
    else:
        n = 1
    
    X = df[in_features].values.reshape((-1,n))
    Y = df[out_feature].values.reshape((-1,1))
    
    return X,Y

def mean_normalize(df, mu=None, sigma=None):
    """
        returns mean normalized dataframe, mean and std for each column in the dataframe
    """
    
    mean = df.mean() if mu is None else mu
    std = df.std() if sigma is None else sigma
    df_norm = (df - mean) / std
    
    return df_norm, mean, std

def copy_dataset(X_tr, Y_tr, X_val, Y_val):    
    return copy(X_tr), copy(Y_tr), copy(X_val), copy(Y_val)


def filter_data(images, labels, classes=[0,1]):
    idx = []
    for i, label in enumerate(labels):
        if label in classes:
            idx.append(i)
    f_images = images[idx]
    f_labels = labels[idx]
    return f_images, f_labels