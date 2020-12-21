"""
Created on Sun Oct 18 14:56:12 2020

@author: aparnami
"""

from itcs4156.datasets.Dataset import Dataset
import os
import pandas as pd
import numpy as np
from struct import unpack

class MNISTDataset(Dataset):

    def __init__(self): 
        
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST")

        self.data = {
           
            "urls" : {
                "train" : "https://drive.google.com/uc?export=download&id=1PepMZ-2uHWf0HO-PG9we03jJ46BRHNUJ",
                "val"  : "https://drive.google.com/uc?export=download&id=1ER4qAUWncgZLSfGL_-hKmMqhFbUaImYt"
            },

            "paths" : {
                "X_train" : os.path.join(self.data_dir, 'train_images.csv'),
                "Y_train" : os.path.join(self.data_dir, 'train_labels.csv'),
                "X_val" : os.path.join(self.data_dir, 'val_images.csv'),
                "Y_val" : os.path.join(self.data_dir, 'val_labels.csv')
            }
        }

        self.init_download()

    def init_download(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for key, url in self.data["urls"].items():
            data_path = self.download(url, self.data_dir, key + '.zip')
            self.extract_zip(data_path, location=self.data_dir)

    def load(self):
        print("Loading dataset..")
        X_train = np.genfromtxt(self.data["paths"]['X_train'], delimiter=',', dtype=np.uint8)
        Y_train = np.genfromtxt(self.data["paths"]['Y_train'], delimiter=',', dtype=np.uint8)
        X_val = np.genfromtxt(self.data["paths"]['X_val'], delimiter=',', dtype=np.uint8)
        Y_val = np.genfromtxt(self.data["paths"]['Y_val'], delimiter=',', dtype=np.uint8) 
        print("Done!")
        Y_train = Y_train.reshape(-1,1)
        Y_val = Y_val.reshape(-1,1)
        return (X_train, Y_train), (X_val, Y_val)
      

if __name__ == "__main__":
    MNISTDataset()