# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random forest classifier model using random state of 42, the remaining parameters are the defaults in RandomForestClassifier of sklearn.ensemble.

## Intended Use
For classifying salaries into two groups: >50K and <=50K.  
Target label is binerized, so that >50K is 1 and <=50K is 0.

## Training Data
Data is census.csv from the 1994 Census database [Data link](https://archive.ics.uci.edu/ml/datasets/census+income).  
80% of this data is used for training (train and validation set).

## Evaluation Data
20% of the data from census.csv were kept aside for evaluation (test set).

## Metrics
Computed metrics are precision, recall, and f1 scores. 
Computed overal scores on the test set: 0.72 0.63 0.67.

## Ethical Considerations
Metrics are also computed for data slices of categorical features.  
It seems that for some race and native country categories, the metrics are much lower than the overal metrics calculated for all data.

## Caveats and Recommendations
Unit tests are not fully implemented.  
This model still needs hyper parameter tuning to improve the metrics.  
