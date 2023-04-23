import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

import plotclassification

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

model = pickle.load(open('/home/bilbo/dev/python/audio_ml/audioml/data/classification.model', 'rb'))
y_test = pickle.load(open('/home/bilbo/dev/python/audio_ml/audioml/data/y_test', 'rb'))
X_test = pickle.load(open('/home/bilbo/dev/python/audio_ml/audioml/data/X_test', 'rb'))
y_pred = model.predict(X_test)

y_preds = get_preds(y_pred)

cr = classification_report(y_test, y_preds)

print(cr)

cf = confusion_matrix(y_test, y_preds)
print(cf)

print("Accuracy:", metrics.accuracy_score(y_test, y_preds))
print("Precision:", metrics.precision_score(y_test, y_preds,average='macro'))
print("Recall:", metrics.recall_score(y_test, y_preds,average='macro'))