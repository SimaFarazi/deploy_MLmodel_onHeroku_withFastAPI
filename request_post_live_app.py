import requests
import json

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

URL_HEROKU = "https://deploy-api-project.herokuapp.com/inference" 
URL_LOCAL = "http://127.0.0.1:8000/inference/"
response = requests.post(URL_HEROKU, data = json.dumps(inference_data_1))
print(f"status code: {response.status_code}")
if(response.json()["prediction"]==[1]):
    prediction = "salary>50K"
else:
    prediction = "salary<=50K"

print(prediction)