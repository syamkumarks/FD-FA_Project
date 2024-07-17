import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

df_rm=pd.read_csv("./Radiomics_Features_Train.csv")
# Load the model
model = load_model('feature_extractor_model_27-05-2024_128.h5')

# Load or prepare your test data (replace this with your actual test data)
# X_test = ... 
# y_test = ...
train_data_path = r'./Train_Roi_file'
train_patients = [name for name in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, name))]
file_list = sorted(train_patients)
print(file_list)
data = [np.expand_dims(np.load(f'{train_data_path}/{file}'), axis=-1) for file in file_list]
X_train = np.array(data)
# Ensure X_test is a NumPy array
# X_test = np.array(X_test)

# Make predictions
predictions = model.predict(X_train)

predictions_df = pd.DataFrame(predictions)
predictions_df['Patient_id']=df_rm['Patient_id']
# Save to CSV
predictions_df.to_csv('Deep_Learning_Features_Train.csv', index=False)