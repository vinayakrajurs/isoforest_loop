# Importing required libraries
import pandas as pd
from sklearn.ensemble import IsolationForest

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection
cols = ['Target 1', 'Target 2']

# Looping through each ID, Sub ID, and Present value in the test data
for i, row in test_data.iterrows():
    test_id = row['ID']
    test_sub_id = row['Sub ID']
    test_present = row['Present']
    
    # Filtering the train data for the current ID, Sub ID, and Present value
    train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
    
    # Selecting the data for training the Isolation Forest model
    train_X = train_subset.loc[:, cols].values
    
    # Creating an instance of Isolation Forest and fitting it with the data
    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)
    clf.fit(train_X)
    
    # Predicting the anomalies for the current data point in the test dataset
    test_X = row[cols].values.reshape(1, -1)
    pred = clf.predict(test_X)
    
    # Outputting the anomalies
    if pred == -1:
        print(f"Anomaly detected for ID {test_id}, Sub ID {test_sub_id}, Present {test_present}")

