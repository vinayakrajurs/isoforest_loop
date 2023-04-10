import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Define target columns to loop through
target_cols = ['target_col_1', 'target_col_2']

# Create empty DataFrame for results
result_df = pd.DataFrame(columns=['Date', 'ID', 'Sub ID', 'Target Col', 'Prediction'])

# Loop through target columns
for col in target_cols:
    # Train Isolation Forest model on train data
    model = IsolationForest()
    model.fit(train_df[[col]])
    
    # Make predictions on test data
    preds = model.predict(test_df[[col]])
    
    # Create DataFrame for predictions and append to results DataFrame
    pred_df = pd.DataFrame({
        'Prediction': preds,
        'Date': test_df['Date'],
        'ID': test_df['ID'],
        'Sub ID': test_df['Sub ID']
    })
    pred_df['Target Col'] = col
    result_df = result_df.append(pred_df)
    
# Save results to new CSV file
result_df.to_csv('result.csv', index=False)
