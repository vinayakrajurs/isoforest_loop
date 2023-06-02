import smtplib
from email.mime.text import MIMEText
from tabulate import tabulate
import pandas as pd

# email details
sender_email = 'your_email@example.com'
sender_password = 'your_email_password'
recipient_email = 'recipient_email@example.com'
email_subject = 'CSV table in email body'
email_body = 'Please find the CSV table below:\n\n'

# read csv file
csv_file_path = 'path/to/your/csv/file.csv'
df = pd.read_csv(csv_file_path)

# create table from csv data
table = tabulate(df, headers='keys', tablefmt='html')

# add table to email body
email_body += f'<pre>{table}</pre>'

# create message object
message = MIMEText(email_body, 'html')
message['From'] = sender_email
message['To'] = recipient_email
message['Subject'] = email_subject

# send email
with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
    smtp.starttls()
    smtp.login(sender_email, sender_password)
    smtp.send_message(message)

#####################################################################
#Grid Search LOF
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Target 1', 'Target 2']

# Filtering the train data for the columns of interest
train_X = train_data[cols].values

# Creating an instance of Local Outlier Factor
clf = LocalOutlierFactor()

# Defining the hyperparameter grid to search
param_grid = {
    'n_neighbors': [5, 10, 15, 20],
    'contamination': [0.01, 0.05, 0.1, 0.15]
}

# Creating an instance of GridSearchCV and fitting it with the data
grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1')
grid_search.fit(train_X)

# Printing the best hyperparameters and the corresponding score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

............................................................................

#Randomsearch lof
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Target 1', 'Target 2']

# Filtering the train data for the columns of interest
train_X = train_data[cols].values

# Creating an instance of Local Outlier Factor
clf = LocalOutlierFactor()

# Defining the hyperparameter distribution to sample from
param_distributions = {
    'n_neighbors': [5, 10, 15, 20],
    'contamination': [0.01, 0.05, 0.1, 0.15]
}

# Creating an instance of RandomizedSearchCV and fitting it with the data
random_search = RandomizedSearchCV(clf, param_distributions=param_distributions, scoring='f1', n_iter=10, random_state=42)
random_search.fit(train_X)

# Printing the best hyperparameters and the corresponding score
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)

#More para
param_distribution = {
    'n_neighbors': [5, 10, 20],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50],
    'p': [1, 2],
}
...............................................................................................
#random search one class svm 
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Setting up the parameter grid for hyperparameter tuning
param_distribution = {'nu': np.linspace(0.01, 0.1, 10),
                      'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto']}

# Function to process each row of the test data
def process_row(row):
    test_id = row['ID']
    test_sub_id = row['Sub ID']
    test_present = row['Present']

    # Filtering the train data for the current ID, Sub ID, and Present value
    train_subset = train_data[(train_data['ID'] == test_id) & (train_data['Sub ID'] == test_sub_id) & (train_data['Present'] == test_present)]
    train_X = train_subset.loc[:, target_cols].values

    # Setting up the One-Class SVM model with hyperparameter tuning using RandomizedSearchCV
    model = OneClassSVM()
    random_search = RandomizedSearchCV(model, param_distribution, cv=5, n_iter=20, n_jobs=-1, random_state=42)
    random_search.fit(train_X)

    # Predicting the anomaly score for the current data point in the test dataset
    test_X = row[target_cols].values.reshape(1, -1)
    pred = random_search.predict(test_X)

    # Returning the anomaly score for the current row
    return pred[0]

# Processing each row of the test data and storing the results in a list
anomaly_scores = []
for i, row in test_data.iterrows():
    score = process_row(row)
    anomaly_scores.append(score)

# Adding the anomaly scores to the test data as a new column
test_data['anomaly_score'] = anomaly_scores

# Saving the test data with anomaly scores to a new CSV file
test_data.to_csv('test_with_scores.csv', index=False)
........................................................
#Grid search one class svm 
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool, cpu_count

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Defining the parameter grid for hyperparameter tuning
param_grid = {'kernel': ['linear', 'rbf', 'sigmoid'], 
              'nu': [0.01, 0.1, 0.5, 0.9]}

# Creating an instance of GridSearchCV for hyperparameter tuning
clf = GridSearchCV(estimator=OneClassSVM(), param_grid=param_grid, cv=5, n_jobs=-1)

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

        # Selecting the data for training the One-class SVM model
        train_X = train_subset.loc[:, target_cols].values

        # Fitting the One-class SVM model with the data
        clf.fit(train_X)

        # Predicting the anomalies for the current data point in the test dataset
        test_X = row[target_cols].values.reshape(1, -1)
        pred = clf.predict(test_X)

        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        new_row['anomaly'] = pred[0]

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
.....................................................................................................
#Grid search isolation forest
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import numpy as np

# define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_samples': [0.25, 0.5, 0.75],
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [1, 2, 3],
}

# create IsolationForest instance
isof = IsolationForest()

# create GridSearchCV object
grid_search = GridSearchCV(isof, param_grid, cv=5, n_jobs=-1)

# fit the model on training data
grid_search.fit(X_train)

# print best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
..........................................................................................................................
#random search isolation forest
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# define parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(100, 1001, 100),
    'max_samples': np.arange(0.1, 1.1, 0.1),
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [1, 2, 3],
}

# create IsolationForest instance
isof = IsolationForest()

# create RandomizedSearchCV object
random_search = RandomizedSearchCV(isof, param_distributions=param_dist, cv=5, n_iter=10, random_state=42, n_jobs=-1)

# fit the model on training data
random_search.fit(X_train)

# print best parameters and score
print("Best parameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)
..............................................................................................................
#Multivariate weighted approch 3 models
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Looping through each row in the test dataset
anomalies = []
for i, row in test_data.iterrows():
    
    test_X = row[target_cols].values.reshape(1, -1)
    
    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)
    iso_forest.fit(train_data[target_cols])
    iso_score = iso_forest.score_samples(test_X)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit(train_data[target_cols])
    lof_score = lof.score_samples(test_X)
    
    # One Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
    ocsvm.fit(train_data[target_cols])
    ocsvm_score = ocsvm.score_samples(test_X)
    
    # Combining the anomaly scores from the three models with custom weights
    combined_score = 0.5*iso_score - 0.2*lof_score - 0.3*ocsvm_score
    
    # Adding the anomaly score to the list of anomalies
    anomalies.append(combined_score[0])

# Adding the list of anomalies to the test dataset as a new column
test_data['anomaly'] = anomalies

# Saving the result DataFrame to a new CSV file
test_data.to_csv('result.csv', index=False)
....................................................................................................................
#atleast 2 columns as anomaly in uni
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
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
        anomalies = []
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

        # Creating a new row to add to the result DataFrame
        new_row = row.to_dict()
        new_row['anomaly'] = ', '.join(anomalies)
        
        # Adding the new row to the result DataFrame
        result_data = result_data.append(new_row, ignore_index=True)

    return result_data
..................................................................................
#common name of target variable in uni
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
        
        # Creating an instance of Isolation Forest and fitting it with the data
        clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)
        clf.fit(train_X)
        
        # Predicting the anomalies for the current data point in the test dataset
        test_X = row[target_col]
        pred = clf.predict(test_X.reshape(1, -1))
        
        # Adding the target column name to the list of anomalies if it's predicted as an anomaly
        if pred[0] == -1:
            anomalies.append(target_col)

    # Filtering out the target columns that are predicted as anomalies
    non_anomaly_cols = set(target_cols) - set(anomalies)

    # Creating a new row to add to the result DataFrame
    new_row = row.to_dict()
    new_row['anomaly'] = ', '.join(anomalies)
    new_row['non_anomaly'] = ', '.join(non_anomaly_cols)
    
    # Adding the new row to the result DataFrame
    result_data = result_data.append(new_row, ignore_index=True)
......................................................................................................................
#as seperate function (incomplete)
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import GridSearchCV

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Defining the parameter grids for GridSearchCV
param_grid_if = {'n_estimators': [100, 200], 'contamination': [0.05, 0.1]}
param_grid_lof = {'n_neighbors': [10, 20], 'contamination': [0.05, 0.1]}
param_grid_svm = {'nu': [0.05, 0.1], 'gamma': [0.05, 0.1]}

# Function to train and predict using Isolation Forest
def run_if(train_X, test_X):
    clf = GridSearchCV(IsolationForest(max_samples='auto', random_state=42), param_grid_if)
    clf.fit(train_X)
    pred = clf.predict(test_X)
    return pred

# Function to train and predict using Local Outlier Factor
def run_lof(train_X, test_X):
    clf = GridSearchCV(LocalOutlierFactor(), param_grid_lof)
    clf.fit(train_X)
    pred = clf.predict(test_X)
    return pred

# Function to train and predict using One Class SVM
def run_svm(train_X, test_X):
    clf = GridSearchCV(OneClassSVM(kernel='rbf'), param_grid_svm)
    clf.fit(train_X)
    pred = clf.predict(test_X)
    return pred

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
            
            # Selecting the data for training the models
            train_X = train_subset.loc[:, target_col].values.reshape(-1, 1)
            test_X = row[target_col].reshape(-1, 1)
            
            # Training and predicting using Isolation Forest
            pred_if = run_if(train_X, test_X)
            
            # Training and predicting using Local Outlier Factor
            pred_lof = run_lof(train_X, test_X)
            
            # Training and predicting using One Class SVM
            pred_svm = run_svm(train_X, test_X)
            
            # Adding the target column name to the list of anomalies if it's predicted as an anomaly by at least 2 models
            if (pred_if[0] == -1 and pred_lof[0] == -1) or (pred_if[0] == -1 and pred_svm[0] ==
..............................................................................................
#Use grid search each time as seperate function (incomplete)
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV

# Reading the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Defining the columns to use for anomaly detection and the target column names
cols = ['Date', 'ID', 'Sub ID', 'Present']
target_cols = ['Target 1', 'Target 2']

# Function to define and train the models
def train_models(train_data, test_data):
    # Defining the hyperparameters to search over for each model
    isolation_forest_params = {
        'n_estimators': [100, 200, 300],
        'contamination': [0.05, 0.1, 0.15],
        'random_state': [42]
    }

    lof_params = {
        'n_neighbors': [10, 20, 30],
        'contamination': [0.05, 0.1, 0.15]
    }

    one_class_svm_params = {
        'nu': [0.05, 0.1, 0.15],
        'gamma': [0.05, 0.1, 0.15]
    }

    # Creating an instance of GridSearchCV for each model
    clf1 = GridSearchCV(IsolationForest(), isolation_forest_params)
    clf2 = GridSearchCV(LocalOutlierFactor(), lof_params)
    clf3 = GridSearchCV(OneClassSVM(kernel='rbf'), one_class_svm_params)

    # Fitting each model with the data
    clf1.fit(train_data)
    clf2.fit(train_data)
    clf3.fit(train_data)

    # Predicting the anomalies for the current data point in the test dataset using Isolation Forest
    pred1 = clf1.predict(test_data)
    
    # Predicting the anomalies for the current data point in the test dataset using Local Outlier Factor
    pred2 = clf2.predict(test_data)
    
    # Predicting the anomalies for the current data point in the test dataset using One Class SVM
    pred3 = clf3.predict(test_data)

    # Adding the target column name to the list of anomalies if it's predicted as an anomaly by at least 2 models
    anomalies = []
    if (pred1[0] == -1 and pred2[0] == -1) or (pred1[0] == -1 and pred3[0] == -1) or (pred2[0] == -1 and pred3[0] == -1):
        anomalies.append(target_col)

    return ', '.join(anomalies)

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
            
.........................................................................................................................................................









