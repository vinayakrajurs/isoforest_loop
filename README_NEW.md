import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Combine train and test data for one-hot encoding
combined_data = pd.concat([train_data, test_data], axis=0)

# Perform one-hot encoding for categorical columns
categorical_columns = ['target1', 'target2']
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = ohe.fit_transform(combined_data[categorical_columns])
encoded_columns = ohe.get_feature_names(categorical_columns)
combined_data.drop(categorical_columns, axis=1, inplace=True)
combined_data[encoded_columns] = encoded_features

# Split the combined data back into train and test sets
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

# Normalize the numeric columns
numeric_columns = ['target1', 'target2']  # Adjust the columns as needed
scaler = StandardScaler()
train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

# Train the Isolation Forest model
model = IsolationForest(contamination=0.01)  # Adjust the contamination parameter as needed
model.fit(train_data.drop('date', axis=1))  # Drop the 'date' column from training data if not needed

# Predict anomalies on the test data
test_data['anomaly_score'] = model.decision_function(test_data.drop('date', axis=1))  # Drop the 'date' column from test data if not needed
test_data['anomaly_label'] = model.predict(test_data.drop('date', axis=1))  # Drop the 'date' column from test data if not needed

# Output the test data with anomaly scores and labels
print(test_data[['date', 'target1', 'target2', 'anomaly_score', 'anomaly_label']])

####################################################################################
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
            
            
#######################################################################################
In this example, the logging module is imported, and the logger is configured using basicConfig(). You can customize the log level (e.g., logging.INFO, logging.DEBUG) and format to suit your needs.

The functions load_data(), preprocess_data(), train_model(), and evaluate_model() represent the different steps of your machine learning pipeline. You can modify them accordingly.

The main() function orchestrates the entire process, calling the necessary functions and handling any exceptions that might occur. The logger is used to log informative messages at various stages of the process.

This code provides a basic structure for incorporating logging into your machine learning model. You can customize it further by adding more logging statements or adjusting the log level as required.

In this updated code, the filename parameter is added to the basicConfig() function, specifying the name of the log file as 'model.log'. Additionally, the filemode parameter is set to 'w' to overwrite the file if it already exists. Adjust these parameters according to your requirements.

When you run this code, the log messages will be saved to the specified log file (model.log in this case). You can access the log file using any text editor or file viewer of your choice to view the logged messages.
