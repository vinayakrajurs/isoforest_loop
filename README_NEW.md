import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import configparser

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def run_anomaly_detection(config):
    # Reading the train and test datasets
    train_data = pd.read_csv(config['train_data']['file'])
    test_data = pd.read_csv(config['test_data']['file'])
    
    # Extracting parameters from the config
    cols = [value for key, value in config['columns'].items()]
    target_cols = [value for key, value in config['target_columns'].items()]
    anomaly_rows = [dict(row.split(', ') for row in value.split('\n')) for key, value in config['anomaly_rows'].items()]
    csv_filename = config['output']['file']
    
    # Creating a list to store the results
    results = []
    
    # Looping through each row in the test data
    for i, row in test_data.iterrows():
        # Getting the ID, Sub ID, and Present value for the current row
        test_id = row['ID']
        test_sub_id = row['Sub ID']
        test_present = row['Present']
        
        # Filtering the train data for the current ID, Sub ID, and Present value
        train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
        
        # Looping through each target column
        anomalies = []
        for target_col in target_cols:
            # Selecting the data for training the Isolation Forest model
            train_X = train_subset.loc[:, target_col].values.reshape(-1, 1)
            
            # Creating an instance of Isolation Forest and fitting it with the data
            clf1 = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)
            clf1.fit(train_X)
            
            # Creating an instance of Local Outlier Factor and fitting it with the data
            clf2 = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            clf2.fit(train_X)
            
            # Creating an instance of One Class SVM and fitting it with the data
            clf3 = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
            clf3.fit(train_X)
            
            # Predicting the anomalies for the current data point in the test dataset using Isolation Forest
            test_X = row[target_col]
            pred1 = clf1.predict(test_X.reshape(1, -1))
            
            # Predicting the anomalies for the current data point in the test dataset using Local
