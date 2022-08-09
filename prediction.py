import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def label_encoder(input_val,le):
    input_val = le.fit_transform(input_val.astype('str'))
    return input_val


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
