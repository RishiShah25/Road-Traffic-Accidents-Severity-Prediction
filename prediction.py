import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def label_encoder(data,le): 
    return data.apply(le.transform)


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
