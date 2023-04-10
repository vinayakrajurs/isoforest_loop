# isoforest_loop
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Define columns to drop
drop_cols = ["date", "ID", "Sub ID"]

# Loop through each target column
for target_col in ["target_1", "target_2"]:
    # Extract feature and target columns
    train_X = train_data.drop(drop_cols + [target_col], axis=1)
    train_y = train_data[target_col]
    test_X = test_data.drop(drop_cols + [target_col], axis=1)

    # Train Isolation Forest model
    clf = IsolationForest(random_state=42, contamination=0.1)
    clf.fit(train_X)

    # Predict on test data
    test_y_pred = clf.predict(test_X)
    test_y_pred = np.where(test_y_pred == -1, 1, 0)  # Convert to 1 for anomaly and 0 for normal

    # Add prediction column to test dataset
    test_data[target_col+"_pred"] = test_y_pred

    # Save predictions to CSV
    test_data.to_csv("test_predictions_{}.csv".format(target_col), index=False)

