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

DATA_PATH = "/home/bilbo/dev/python/audio_ml/audioml/data/speakers_1.json"


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


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def show_model(model):
    """
    plot_model(model,
    to_file='model.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=False,
    rankdir='TB')
    """
    plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
    show_trainable=False
    )


def build_model(input_shape):
    """Generates RNN-LSTM model

    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(11, activation='softmax'))  # the number is the number of categories

    print(model.summary())
    return model

def get_preds(y_pred):
    preds = []
    for i in range(0,len(y_pred)):
        max_p = -1
        max_i = -1
        for index, prob in enumerate(y_pred[i]):
            if prob > max_p:
                max_p = prob
                max_i = index
        preds.append(max_i)
    return preds


def train(best_score):
    

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    y_test_df = pd.DataFrame(y_test,columns=['classification'])
    

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)
    show_model(model)
    

    # compile modelshow_
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)
 
    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    if test_acc > best_score:
        
        best_score = test_acc        
        pickle.dump(model, open('/home/bilbo/dev/python/audio_ml/audioml/data/classification.model', 'wb'))
        pickle.dump(y_test, open('/home/bilbo/dev/python/audio_ml/audioml/data/y_test', 'wb'))
        pickle.dump(X_test, open('/home/bilbo/dev/python/audio_ml/audioml/data/X_test', 'wb'))
        
        y_pred = model.predict(X_test)
        y_preds = get_preds(y_pred)
        cr = classification_report(y_test, y_preds)
        print(cr)
        print("Accuracy:", metrics.accuracy_score(y_test, y_preds))
        print("Precision:", metrics.precision_score(y_test, y_preds,average='macro'))
        print("Recall:", metrics.recall_score(y_test, y_preds, average='macro'))

        # plot accuracy/error for training and validation
        #plot_history(history)
    return best_score

    
if __name__ == "__main__":
    

    best_score = 0
    for i in range(0,10):
        best_score = train(best_score)   
        print(f"trainng --- best score so far -- {best_score}")
        evaluate.evaluate_model()
    
