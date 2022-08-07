import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def label_encoder(data,le): 
    for feat in data:
        data[feat] = data[feat].astype(str)
        data[feat] = le.fit_transform(data[feat])
    return data


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
