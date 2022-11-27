from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from data import process_data
import joblib


# Optional: implement hyperparameter tuning.
def train_model_rf(X_train, y_train):
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
        Trained machine learning model (Random Forest).
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    return model


def compute_cross_val_score(model, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model : 
        Trained machine learning model.
    Returns
    -------
    Scores : np.array
        Scores for each fold 
    """
    # calculate scores for 5-folds cross validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')
    return scores


def cross_validation(model, X_train, y_train, cv=5):
    """
    Function to perform 5 folds cross-validation parameters.
    https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/
    Inputs
    ------
    model: 
        This is the machine learning algorithm to be used for training.
    X_train: array
        This is the matrix of features.
    y_train: array
        This is the target variable.
    cv: int, default=5
        Number of folds for cross-validation.
    Returns
    -------
        A dictionary containing the metrics 'accuracy', 'precision',
        'recall', 'f1' for both training set and validation set.
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv, scoring=scoring, return_train_score=True)
    return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
    }

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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
    preds = model.predict(X)
    return preds

def compute_model_metrics_slice(data, cat_feat_name, unique_cat):
    """ Compute model metrics for each unique category of a categorical feature.

    Inputs
    ------
    data: data frame
        Test set data frame for validation
    cat_feat_name :
        Name of the categorical feature.
    unique_cat :
        Name of unique category in cat_feat
    Returns
    -------
    metric_list: list
    [
        precision : float
        recall : float
        fbeta : float
    ]
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
    # load encoder and binerizer from model directory
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    rf_model = joblib.load("model/random_forest_model.pkl")

    # Proces a slice of test data with the process_data function.
    X_test_slice, y_test_slice, encoder_slice, lb_slice = process_data(
    data[data[cat_feat_name]==unique_cat], 
    categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
    )
    precision_slice, recall_slice, fbeta_slice = compute_model_metrics(y_test_slice, rf_model.predict(X_test_slice))
    metric_list = [precision_slice, recall_slice, fbeta_slice]
    return metric_list
