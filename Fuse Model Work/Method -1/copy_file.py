import os
import shutil
import csv

def copy_files_with_suffix(source_dir, destination_dir_image, destination_dir_mask, image_suffix,mask_suffix):
    # Ensure the destination directory exists
    os.makedirs(destination_dir_image, exist_ok=True)
    os.makedirs(destination_dir_mask, exist_ok=True)
    # Iterate through all files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file name ends with the specified suffix
            print(file.split("_"))
            if file[-8:]==image_suffix:
                source_image_path = os.path.join(root, file)
                destination_image_path = os.path.join(destination_dir_image, file)
                # Copy the file to the destination directory
                shutil.copy2(source_image_path, destination_image_path)
                print(f"Copied: {file} from {source_image_path} to {destination_image_path}")
                image_file=file
                data=[file.split("_")[0],file.split("_")[1],image_file,None]
                with open('example_with_mask.csv', 'a', newline='') as file1:
                    writer = csv.writer(file1)
                    writer.writerow(data)
            if file[-10:]==mask_suffix:
                source_mask_path = os.path.join(root, file)
                destination_mask_path = os.path.join(destination_dir_mask, file)
                # Copy the file to the destination directory
                shutil.copy2(source_mask_path, destination_mask_path)
                print(f"Copied: {file} from {source_mask_path} to {destination_mask_path}")
                mask_file=file
            

# Replace these paths and pattern with your actual values
source_directory = '../../DataSet Files/Test_Files'
destination_directory_image = './ROI_ny_files'
destination_directory_mask = './Mask_ny_files'
image_name_suffix = '_roi.npy'
mask_name_suffix = '_label.npy'

copy_files_with_suffix(source_directory, destination_directory_image,destination_directory_mask, image_name_suffix,mask_name_suffix)
