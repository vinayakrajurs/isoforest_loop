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


#END of code
#ALL 3 MODELS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# function to apply Isolation Forest anomaly detection model
def apply_isolation_forest(df_train, df_test):
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(df_train)
    y_pred = clf.predict(df_test)
    return y_pred

# function to apply LOF anomaly detection model
def apply_lof(df_train, df_test):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit(df_train)
    y_pred = clf.fit_predict(df_test)
    return y_pred

# function to apply One Class SVM anomaly detection model
def apply_one_class_svm(df_train, df_test):
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(df_train)
    y_pred = clf.predict(df_test)
    return y_pred

# function to get anomalies and return result dataframe
def get_anomalies(model_func, train_path, test_path, result_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_test["anomaly"] = model_func(df_train, df_test.iloc[:,1:-1])
    df_test.to_csv(result_path, index=False)

# paths to train, test and result files
train_path = "train_data.csv"
test_path = "test_data.csv"
result_path_iforest = "results_isolation_forest.csv"
result_path_lof = "results_lof.csv"
result_path_svm = "results_one_class_svm.csv"
result_path_concat = "results_concatenated.csv"

# apply isolation forest and save results
get_anomalies(apply_isolation_forest, train_path, test_path, result_path_iforest)

# apply LOF and save results
get_anomalies(apply_lof, train_path, test_path, result_path_lof)

# apply One Class SVM and save results
get_anomalies(apply_one_class_svm, train_path, test_path, result_path_svm)

# concatenate anomaly column from all 3 models and save final results
df_iforest = pd.read_csv(result_path_iforest)
df_lof = pd.read_csv(result_path_lof)
df_svm = pd.read_csv(result_path_svm)

df_concat = pd.concat([df_iforest["anomaly"], df_lof["anomaly"], df_svm["anomaly"]], axis=1)
df_concat.columns = ["anomaly_iforest", "anomaly_lof", "anomaly_svm"]
df_concat.to_csv(result_path_concat, index=False)



##GRID SEARCH ISOLATION FOREST
import pandas as pd
from sklearn.ensemble import IsolationForest
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import GridSearchCV

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
        for target_col in target_cols:
            # Filtering the train data for the current ID, Sub ID, and Present value
            train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
            
            # Selecting the data for training the Isolation Forest model
            train_X = train_subset.loc[:, target_col].values.reshape(-1, 1)
            
            # Tuning the hyperparameters of Isolation Forest using Grid Search
            params = {'n_estimators': [50, 100, 200], 'contamination': [0.05, 0.1, 0.2]}
            clf = GridSearchCV(IsolationForest(max_samples='auto', random_state=42), params, cv=5)
            clf.fit(train_X)
            
            # Predicting the anomalies for the current data point in the test dataset
            test_X = row[target_col]
            pred = clf.predict(test_X.reshape(1, -1))
            
            # Adding the target column name to the list of anomalies if it's predicted as an anomaly
            if pred[0] == -1:
                anomalies.append(target_col)

        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        new_row['anomaly'] = ', '.join(anomalies)
        
        # Adding the new row to the result DataFrame
        result_data = result_data.append(new_row, ignore_index=True)

    return result_data


# Dividing the test data into equal chunks (except for the last one)
chunk_size = len(test_data) // cpu_count()
test_data_chunks = [test_data.iloc[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]

# Processing the chunks in parallel using multiprocessing
with Pool(cpu_count()) as pool:
    result_chunks = pool.map(process_chunk, test_data_chunks)

# Combining the results into one DataFrame
result_data = pd.concat(result_chunks, ignore_index=True)

# Saving the result DataFrame to a new CSV file
result_data.to_csv('result.csv', index=False)



##ISOLATION FOREST MULTIVARIATE
import pandas as pd
from sklearn.ensemble import IsolationForest
from multiprocessing import Pool, cpu_count

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
        
        # Filtering the train data for the current ID, Sub ID, and Present value
        train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
        
        # Selecting the data for training the Isolation Forest model
        train_X = train_subset.loc[:, target_cols].values
        
        # Creating an instance of Isolation Forest and fitting it with the data
        clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)
        clf.fit(train_X)
            
        # Predicting the anomalies for the current data point in the test dataset
        test_X = row[target_cols].values.reshape(1, -1)
        pred = clf.predict(test_X)
        
        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        for j, target_col in enumerate(target_cols):
            # Adding the target column name to the list of anomalies if it's predicted as an anomaly
            if pred[0][j] == -1:
                new_row[target_col+'_anomaly'] = 1
            else:
                new_row[target_col+'_anomaly'] = 0
        
        # Adding the new row to the result DataFrame
        result_data = result_data.append(new_row, ignore_index=True)

    return result_data


# Dividing the test data into equal chunks (except for the last one)
chunk_size = len(test_data) // cpu_count()
test_data_chunks = [test_data.iloc[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]

# Processing the chunks in parallel using multiprocessing
with Pool(cpu_count()) as pool:
    result_chunks = pool.map(process_chunk, test_data_chunks)

# Combining the results into one DataFrame
result_data = pd.concat(result_chunks, ignore_index=True)

# Saving the result DataFrame to a new CSV file
result_data.to_csv('result.csv', index=False)



##LOF MULTIVARIATE
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import Pool, cpu_count

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
        anomalies = {}
        for target_col in target_cols:
            # Filtering the train data for the current ID, Sub ID, and Present value
            train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
            
            # Selecting the data for training the Local Outlier Factor model
            train_X = train_subset.loc[:, target_cols].values
            
            # Creating an instance of Local Outlier Factor and fitting it with the data
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            clf.fit(train_X)
            
            # Predicting the anomalies for the current data point in the test dataset
            test_X = row[target_cols].values.reshape(1, -1)
            pred = clf.predict(test_X)
            
            # Adding the target column name to the list of anomalies if it's predicted as an anomaly
            if pred[0] == -1:
                anomalies[target_col] = True
            else:
                anomalies[target_col] = False

        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        new_row.update(anomalies)
        
        # Adding the new row to the result DataFrame
        result_data = result_data.append(new_row, ignore_index=True)

    return result_data


# Dividing the test data into equal chunks (except for the last one)
chunk_size = len(test_data) // cpu_count()
test_data_chunks = [test_data.iloc[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]

# Processing the chunks in parallel using multiprocessing
with Pool(cpu_count()) as pool:
    result_chunks = pool.map(process_chunk, test_data_chunks)

# Combining the results into one DataFrame
result_data = pd.concat(result_chunks, ignore_index=True)

# Saving the result DataFrame to a new CSV file
result_data.to_csv('result.csv', index=False)

