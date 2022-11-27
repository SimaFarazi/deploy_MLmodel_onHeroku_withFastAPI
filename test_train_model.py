import os
import pandas as pd

ENCODER_PATH = "model/encoder.joblib"
LB_PATH = "model/lb.joblib"
MODEL_PATH = "model/random_forest_model.pkl"
SLICE_METRIC_PATH = "./slice_output.txt"

def test_white_space_removal():
    data = pd.read_csv("data/census.csv", skipinitialspace=True)
    assert [str.isspace()==False for str in data.columns.values.tolist()], "White space in column names!"

def test_process_data_output():
    assert os.path.isfile(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0, "Encoder.joblib is not available!"
    assert os.path.isfile(LB_PATH) and os.path.getsize(LB_PATH) > 0, "lb.joblib is not available!"

def test_train_model_output():
    assert os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0, "random_forest_model.pkl is not available!"

def test_compute_metric_slice_output():
    assert os.path.isfile(SLICE_METRIC_PATH) and os.path.getsize(SLICE_METRIC_PATH) > 0, "slice_output.txt is not available!"