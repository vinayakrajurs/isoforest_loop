import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define columns to use for isolation forest
target_cols = ['Target 1', 'Target 2']

# Iterate over target columns
for target_col in target_cols:
    # Define numeric columns to use for isolation forest
    num_cols = [col for col in train_data.columns if train_data[col].dtype != 'object' and col != target_col]
    
    # Fit isolation forest on training data
    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, random_state=0)
    clf.fit(train_data[num_cols])
    
    # Predict anomalies on test data
    test_data[target_col] = clf.predict(test_data[num_cols])

# Save results to csv
test_data.to_csv('anomaly_results.csv', index=False)
