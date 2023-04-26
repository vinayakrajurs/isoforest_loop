#Isolation Forest
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Creating a new DataFrame to store the results
result_data = pd.DataFrame(columns=test_data.columns)

# Looping through each row in the test data
for i, row in test_data.iterrows():
    test_id = row['ID']
    test_sub_id = row['Sub ID']
    test_present = row['Present']
    
    # Looping through each target column
    for target_col in target_cols:
        # Filtering the train data for the current ID, Sub ID, and Present value
        train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
        
        # Selecting the data for training the LOF model
        train_X = train_subset.loc[:, target_col].values.reshape(-1, 1)
        
        # Creating an instance of LOF and fitting it with the data
        clf = LocalOutlierFactor(n_neighbors=20, contamination=float(0.1))
        clf.fit(train_X)
        
        # Predicting the anomalies for the current data point in the test dataset
        test_X = row[target_col]
        pred = clf.predict(test_X.reshape(1, -1))
        
        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        if pred == -1:
            new_row['anomaly'] = target_col
        else:
            new_row['anomaly'] = ''
        
        # Adding the new row to the result DataFrame
        result_data = result_data.append(new_row, ignore_index=True)

# Saving the result DataFrame to a new CSV file
result_data.to_csv('result.csv', index=False)
