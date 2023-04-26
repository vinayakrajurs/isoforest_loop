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



## modle 2
import pandas as pd
from sklearn.svm import OneClassSVM

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
    anomalies = []
    for target_col in target_cols:
        # Filtering the train data for the current ID, Sub ID, and Present value
        train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
        
        # Selecting the data for training the One-Class SVM model
        train_X = train_subset.loc[:, target_col].values.reshape(-1, 1)
        
        # Creating an instance of One-Class SVM and fitting it with the data
        clf = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
        clf.fit(train_X)
        
        # Predicting the anomalies for the current data point in the test dataset
        test_X = row[target_col]
        pred = clf.predict(test_X.reshape(1, -1))
        
        # If the predicted value is -1, it is an anomaly for this target column
        if pred == -1:
            anomalies.append(target_col)
        
    # Creating a new row to add to the result DataFrame
    new_row = row.to_dict()
    if len(anomalies) > 0:
        new_row['anomaly'] = ','.join(anomalies)
    else:
        new_row['anomaly'] = 'None'
    
    # Adding the new row to the result DataFrame
    result_data = result_data.append(new_row, ignore_index=True)

# Saving the result DataFrame to a new CSV file
result_data.to_csv('result.csv', index=False)


##Phrophet FB modrl
# Import required libraries
import pandas as pd
from fbprophet import Prophet
import multiprocessing as mp

# Load train and test time series data into pandas DataFrames
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Rename columns to "ds" and "y" as required by Prophet
train_df = train_df.rename(columns={'timestamp': 'ds', 'value': 'y'})
test_df = test_df.rename(columns={'timestamp': 'ds', 'value': 'y'})

# Create Prophet model and fit the train data
model = Prophet()
model.fit(train_df)

# Generate a dataframe of future timestamps to make predictions on
future = model.make_future_dataframe(periods=len(test_df))

# Use the model to make predictions on the future data
forecast = model.predict(future)

# Merge train and test dataframes for evaluation
merged_df = pd.concat([train_df, test_df], ignore_index=True)

# Split the test data into chunks for parallel processing
num_chunks = mp.cpu_count()  # Number of chunks equals the number of CPU cores
chunk_size = len(test_df) // num_chunks
test_chunks = [test_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
if len(test_df) % num_chunks != 0:
    # Add remaining rows to last chunk
    test_chunks[-1] = pd.concat([test_chunks[-1], test_df.iloc[num_chunks*chunk_size:]])

# Define function for anomaly detection
def detect_anomalies(test_chunk):
    chunk_min = test_chunk['ds'].min()
    chunk_max = test_chunk['ds'].max()
    chunk_forecast = forecast[(forecast['ds'] >= chunk_min) &
                              (forecast['ds'] <= chunk_max)]
    chunk_anomalies = chunk_forecast[((chunk_forecast['yhat_upper'] < test_chunk['y']) |
                                      (chunk_forecast['yhat_lower'] > test_chunk['y']))]
    return chunk_anomalies

# Process test data chunks in parallel using multiprocessing
pool = mp.Pool(num_chunks)
results = pool.map(detect_anomalies, test_chunks)
pool.close()
pool.join()

# Combine results from each chunk into a single DataFrame
anomalies = pd.concat(results, ignore_index=True)

# Print the anomalies
print(anomalies)

