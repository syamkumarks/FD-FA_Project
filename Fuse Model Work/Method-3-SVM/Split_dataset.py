import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
df = pd.read_csv('./Radiomics_Feature_Augmented.csv')

# Display the first few rows of the dataframe (optional)
print(df.head())

# Define the test set size as an absolute number
test_size_absolute = 1000 # Replace with your desired test set size

# Calculate the test size as a fraction of the total dataset size
test_size_fraction = test_size_absolute / len(df)
test_size_fraction=0.2
# Split the dataframe into training and testing sets
train_df, test_df = train_test_split(df, test_size=test_size_fraction, random_state=42)
sort_column='Patient_id'
# Display the shapes of the resulting dataframes (optional)
print('Training set shape:', train_df.shape)
print('Testing set shape:', test_df.shape)
train_df.sort_values(by=sort_column)
test_df.sort_values(by=sort_column)
# Save the resulting dataframes to new CSV files
train_df.to_csv('Radiomics_Feature_Train.csv', index=False)
test_df.to_csv('Radiomics_Feature_Test.csv', index=False)

print("Training and testing sets have been saved to 'train_dataset.csv' and 'test_dataset.csv'.")
