import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def label_encoder(input_val,le):
    return le.fit_transform(input_val)


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)
