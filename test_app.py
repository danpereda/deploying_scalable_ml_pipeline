from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.


def test_api_locally_get_root():
    r = client.get("/")
    assert r.json() == {
        "greeting": "This is an example of an scalable ml pipeline!"}
    assert r.status_code == 200


def test_predict_bad_input():
    response = client.post(
        "predict/",
        json={
            "age": "-",
            "workclass": " Federal-gov",
            "fnlgt": 176969,
            "education": " HS-grad",
            "education-num": 9,
            "marital-status": " Divorced",
            "occupation": " Prof-specialty",
            "relationship": " Not-in-family",
            "race": " White",
            "sex": " Male",
            "capital-gain": 0,
            "capital-loss": 1590,
            "hours-per-week": 40,
            "native-country": " United-States"
        },
    )
    assert response.status_code == 422


def test_predict_less_than_50k():
    response = client.post(
        "predict/",
        json={
            'age': 34,
            'workclass': ' Private',
            'fnlgt': 198613,
            'education': ' 11th',
            'education-num': 7,
            'marital-status': ' Married-civ-spouse',
            'occupation': ' Sales',
            'relationship': ' Husband',
            'race': ' White',
            'sex': ' Male',
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': 25,
            'native-country': ' ?'
        },
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": " <=50K"}


def test_predict_greater_than_50k():
    response = client.post(
        "predict/",
        json={
            'age': 27,
            'workclass': ' Private',
            'fnlgt': 285897,
            'education': ' HS-grad',
            'education-num': 9,
            'marital-status': ' Married-civ-spouse',
            'occupation': ' Prof-specialty',
            'relationship': ' Husband',
            'race': ' White',
            'sex': ' Male',
            'capital-gain': 0,
            'capital-loss': 1848,
            'hours-per-week': 45,
            'native-country': ' United-States'
        },
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": " >50K"}
