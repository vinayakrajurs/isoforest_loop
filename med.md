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

def record_results(results, file_path):
    # Creating an empty DataFrame to store the final results
    final_results = pd.DataFrame(columns=['ID', 'Sub ID', 'True Positives', 'False Positives', 'Total Anomalies', 'Precision', 'Recall'])

    # Looping through the results from each chunk and recording the true positives and false positives
    for result in results:
        true_positives = 0
        false_positives = 0
        total_anomalies = 0

        for i, row in result.iterrows():
            if row['anomaly'] != '':
                total_anomalies += 1

                if (row['ID'], row['Sub ID'], row['Present']) in anomaly_rows:
                    true_positives += 1
                else:
                    false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(anomaly_rows)

        # Adding the results to the final DataFrame
        final_results = final_results.append({'ID': result['ID'].iloc[0],
                                              'Sub ID': result['Sub ID'].iloc[0],
                                              'True Positives': true_positives,
                                              'False Positives': false_positives,
                                              'Total Anomalies': total_anomalies,
                                              'Precision': precision,
                                              'Recall': recall},
                                             ignore_index=True)

    # Adding the median value of the anomaly target column based on the train data of that target column to the final results
    for col in target_cols:
        train_median = train_data[train_data[col].notna()].groupby(['ID', 'Sub ID', 'Present'])[col].median().reset_index()
        final_results = pd.merge(final_results, train_median, on=['ID', 'Sub ID', 'Present'], how='left')

    # Writing the results to a CSV file
    final_results.to_csv(file_path, index=False)
.....................................................................................................................................
