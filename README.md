# isoforest_loop

import pandas as pd
from sklearn.ensemble import IsolationForest

# read in the train and test CSV files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# specify the date and ID columns
date_col = 'date'
id_col = 'ID'
sub_id_col = 'Sub_ID'

# loop through each target column
target_cols = ['target1', 'target2']
for col in target_cols:

    # create the Isolation Forest model
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.1), random_state=42)

    # fit the model on the train data
    model.fit(train_df.loc[train_df['target']==col, ['value']])

    # use the model to detect anomalies in the test data
    test_df['anomaly'] = model.predict(test_df.loc[test_df['target']==col, ['value']])

    # convert the anomaly scores to binary labels (1 for normal, -1 for anomaly)
    test_df.loc[test_df['target']==col, 'anomaly'] = test_df.loc[test_df['target']==col, 'anomaly'].apply(lambda x: 1 if x == 1 else -1)

    # save the results to a new CSV file with binary labels
    results_df = test_df[test_df['target']==col].copy()
    results_df.loc[:, 'anomaly'] = results_df.loc[:, 'anomaly'].astype(int)
    results_df.to_csv(f"results_{col}.csv", index=False)
