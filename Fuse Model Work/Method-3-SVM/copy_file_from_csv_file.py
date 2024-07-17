import os
import csv
import shutil

def Move_file_from_csv(csv_file,source_folder,destination_folder,column_name):

    # Read the CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            file_name = row[column_name]
            
            # Construct the file name based on the patient ID
                      
            # Construct the source and destination file paths
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            
            # Check if the source file exists
            if os.path.exists(source_file):
                # Move the file from the source folder to the destination folder
                shutil.move(source_file, destination_file)
                print(f"Moved file: {file_name}")
            else:
                print(f"File not found: {file_name}")

csv_file='./Radiomics_Feature_Train.csv'
source_folder='./ROI_ny_files'
destination_folder="Train_Roi_file"
column_name="Patient_id"
Move_file_from_csv(csv_file,source_folder,destination_folder,column_name)
