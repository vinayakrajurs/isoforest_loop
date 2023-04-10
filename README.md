import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Set columns for grouping the data
group_cols = ['day', 'ID', 'Sub ID']

# Create an empty DataFrame to store the results
results_df = pd.DataFrame()

# Loop through each unique day value in the train data
for day_type in train_data['day'].unique():
    # Loop through each target variable
    for target_col in ['target_1', 'target_2']:
        # Filter the train and test data by day type and target column
        train_subset = train_data.loc[(train_data['day'] == day_type) & (train_data[target_col].notnull())]
        test_subset = test_data.loc[test_data['day'] == day_type]

        # Separate the features and target variables
        X_train = train_subset.drop(target_col, axis=1)
        y_train = train_subset[target_col]
        X_test = test_subset.drop(target_col, axis=1)

        # Fit the Isolation Forest model to the training data
        model = IsolationForest()
        model.fit(X_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Add the predictions to the test data
        test_subset[target_col] = y_pred

        # Merge the test data with the original data to get the corresponding date, ID, and Sub ID
        test_results = pd.merge(test_subset, test_data[group_cols], on=group_cols)

        # Append the results to the results DataFrame
        results_df = results_df.append(test_results, ignore_index=True)

# Save the results DataFrame to a CSV file
results_df.to_csv("result.csv", index=False)
