# %% Script to train machine learning model.
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
# %% Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.strip()
# %% train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# %% Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# %% Train and save a model.
model = train_model(X_train, y_train)

# save the model to disk
models = [model, encoder, lb]
model_names = ["model", "encoder", "lb"]
for m, name in zip(models, model_names):
    joblib.dump(m, f"../model/{name}.pkl")

# check performance metrics
preds = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision:{precision:%}")
print(f"Recall:{recall:%}")
print(f"fbeta:{fbeta:%}")
