# deploy MLmodel on Heroku with FastAPI
This repository provides required code for third project of MLDevOps course.
## ML model preparation
**data** folder contains the census data set  
**data.py** contains the preprocessing function  
**model.py** contains train, inference, and compute metrics required functions  
**train_model.py** functions from data.py and model.py are called  
**model** folder contains encoder, label binerizer and model files  
**test_model** has 4 unit tests for ml model preparion  

## API preparation with FastAPI
**main.py** provides GET and POST API endpoints, GET for greeting and POST for providing salary prediction from ML model  
**test_main.py** contains 3 unit tests, one for GET method and 2 for POST method for both 0 and 1   predictions

## Deploy API on Heroku
An App is created on Heroku and connected to this git repo  
The App is automatically deployed(CD) after tests are passed in Github Action(CI)  
Corresponding file for Github Action can be found in .github/workflows/PytestFlake8.yml  
**request_post_live.py** contains request.post to test the live App


