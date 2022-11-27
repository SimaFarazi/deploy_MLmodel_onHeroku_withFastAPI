from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_get_method_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greetings": "Welcome to salary predictor app!"}, "wrong message is shown!"

def test_post_method_inference_0():
    # Inference data as a dict (0:<=50K)
    inference_data_0 = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 231569,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    response = client.post("/inference/", json=inference_data_0)
    assert response.status_code == 200, "response not successful with {}".format(response.json())
    assert response.json() == {"prediction":[0]}, "prediction is wrong, '[0]' were expected"

def test_post_method_inference_1():
    # Inference data as a dict (1:>50K)
    inference_data_1 = {
        "age": 46,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 328216,
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States",
    }
    response = client.post("/inference/", json=inference_data_1)
    assert response.status_code == 200, "response not successful with {}".format(response.json())
    assert response.json() == {"prediction":[1]}, "prediction is wrong, '[1]' were expected"