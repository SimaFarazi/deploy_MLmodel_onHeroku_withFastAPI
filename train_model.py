# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model_rf, inference
from model import compute_model_metrics, compute_model_metrics_slice
from model import compute_cross_val_score, cross_validation
import pandas as pd
import joblib
import numpy as np

# Add code to load in the data.
# skipinitialspace=True is required to get rid of all white spaces
data = pd.read_csv("data/census.csv", skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/lb.joblib')

# Proces the test data with the process_data function.
X_test, y_test, encoder_, lb_ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
)

# Train and save random forest model.
rf_model_ = train_model_rf(X_train, y_train)
# Save model in model folder
joblib.dump(rf_model_, 'model/random_forest_model.pkl')

# Load model
rf_model = joblib.load('model/random_forest_model.pkl')
# Calculate preditions on the test set
preds = inference(rf_model, X_test)

# Compute overal model metrics: percision, recall, and f1
print("Overal model performance in predicticting salary >50K:")
print("precision, recall, f1 scores")
print("%.2f %.2f %.2f" % (compute_model_metrics(y=y_test, preds=preds)))

# Perform cross validation and compute precision on each fold
scores_rf = compute_cross_val_score(rf_model, X_train, y_train)
print("Model precision for each fold by 5-folds cross validation:")
print(np.round(scores_rf,2))

# Perform cross validation and compute all metrics on each fold
# Save compued metrics 
metric_dict = cross_validation(rf_model, X_train, y_train)
file = open("model/cross_val_5folds_metrics_rf.json", "w")
file.write("{\n")
for k in metric_dict.keys():
    file.write("'{}':'{}'\n".format(k, metric_dict[k]))
file.write("}")
file.close()

# Compute model metrics on slice of data: percision, recall, and f1
# Save computed metrics
file = open("slice_output.txt", "w")
cat_feat = "education"
for cat_feat_unique in test[cat_feat].unique():
    metric_cat_slice_list = compute_model_metrics_slice(test, cat_feat, cat_feat_unique)
    precision_sl, recall_sl, f1_sl = np.round(metric_cat_slice_list, 2)
    file.write(f"{cat_feat}={cat_feat_unique}:\n precision={precision_sl},recall={recall_sl},f1={f1_sl}")
    file.write("\n")
file.close()

