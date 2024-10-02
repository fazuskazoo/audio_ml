import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
import evaluate

DATA_PATH = "/home/bilbo/dev/misc_audio/inagural/inagural.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_test():
    
    # get train, validation, test splits
    X_test, y_test = load_data(DATA_PATH)

    y_test_df = pd.DataFrame(y_test,columns=['classification'])
               
    pickle.dump(y_test, open('/home/bilbo/dev/misc_audio/inagural/y_test', 'wb'))
    pickle.dump(X_test, open('/home/bilbo/dev/misc_audio/inagural/X_test', 'wb'))
        
           
if __name__ == "__main__":
    
    prepare_test()   
       
