import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Function to process each chunk of the test data in parallel
def process_chunk(chunk):
    # Creating a new DataFrame to store the results for the current chunk
    result_data = pd.DataFrame(columns=test_data.columns)

    # Looping through each row in the chunk
    for i, row in chunk.iterrows():
        test_id = row['ID']
        test_sub_id = row['Sub ID']
        test_present = row['Present']
        
        # Looping through each target column
        anomalies = []
        medians = []
        for target_col in target_cols:
            # Filtering the train data for the current ID, Sub ID, and Present value
            train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
            
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
            
            # Predicting the anomalies for the current data point in the test dataset using Local Outlier Factor
            pred2 = clf2.predict(test_X.reshape(1, -1))
            
            # Predicting the anomalies for the current data point in the test dataset using One Class SVM
            pred3 = clf3.predict(test_X.reshape(1, -1))
            
            # Adding the target column name to the list of anomalies if it's predicted as an anomaly by at least 2 models
            if (pred1[0] == -1 and pred2[0] == -1) or (pred1[0] == -1 and pred3[0] == -1) or (pred2[0] == -1 and pred3[0] == -1):
                anomalies.append(target_col)
                medians.append(train_subset[target_col].median())

        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        new_row['anomaly'] = ', '.join(anomalies)
        new_row['median'] = ', '.join
................................................................................
