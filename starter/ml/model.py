from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from tqdm.auto import trange
from .data import process_data
import pandas as pd
import numpy as np


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, 
    and f1 score.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics_slices(
        model: ClassifierMixin,
        encoder,
        X_test,
        y_test) -> pd.DataFrame:
    """compute model metrics on slices of categorical data

    Args:
        model (ClassifierMixin): sklearn classifier
        df (pd.DataFrame): dataframe used to train and validate model

    Returns:
        pd.DataFrame: metrics on slices sorted by f1_score
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    encoded_feature = encoder.get_feature_names_out()
    n_features = X_test.shape[1]

    # predict for only data where ohe(category_value) == 1
    performance_dict = {}
    for k in trange(6, n_features):
        mask = (X_test[:, k] == 1)
        try:
            preds = model.predict(X_test[mask])
            performance_dict[encoded_feature[k - 6]
                             ] = compute_model_metrics(y_test[mask], preds)
        except ValueError:
            # if that category is not in test, then return nan
            performance_dict[encoded_feature[k - 6]] = (np.nan, np.nan, np.nan)

    performance = pd.DataFrame.from_dict(performance_dict,
                                         orient="index",
                                         columns=["precision", "recall", "f1"])

    # map xk_ to category name
    x2cat = dict(
        zip([f"x{k}" for k in range(len(cat_features))], cat_features))
    # Add category column to performance dataframe
    performance["category"] = performance.index.str.extract(r"(x\d)")[
        0].map(x2cat).values
    performance.index = performance.index.str.replace(
        r"(x\d_ )", "", regex=True)

    performance.rename_axis("subcategory", inplace=True)
    performance.to_csv(r'metric_on_slices.txt', sep=' ', mode='a')
    return performance.sort_values(by="f1", ascending=False)


if __name__ == "__main__":
    pass
