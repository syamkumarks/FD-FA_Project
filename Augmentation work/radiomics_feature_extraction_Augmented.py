# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:44:09 2019

@author: PC
"""

import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import measure


def Img_Normalization(Image_Orig):
    Image_array = sitk.GetArrayFromImage(Image_Orig)
    min_ImgValue = Image_array.min()
    max_ImgValue = Image_array.max()
    ImgRange = max_ImgValue-min_ImgValue
    min_NewValue = 0
    max_NewValue = 1200
    NewRange = max_NewValue-min_NewValue
    Img_array = ((Image_array-min_ImgValue)/ImgRange)*NewRange+min_NewValue
    Img = sitk.GetImageFromArray(Img_array.astype(int))
    Img.SetDirection(Image_Orig.GetDirection())
    Img.SetOrigin(Image_Orig.GetOrigin())
    Img.SetSpacing(Image_Orig.GetSpacing())
#    Img.CopyInformation(Image_Orig)
    return Img
    
def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    #start = time.clock()
    warnings.simplefilter('ignore')

    train_data_path = r'../DataSet Files/Nodule_crop'
    #list_path = r'.\GGO_DataSet\test_data\test.csv'
    error_file="Error_list_radiomics.txt"
    label_file="GFG_with_new_data.csv"
    # f = open(list_path)
    data = pd.read_csv(label_file)
  
    train_patients = [os.path.join(train_data_path,name) for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]
    train_patients = sorted(train_patients)
    print(train_patients)
    Feature = []
    
    for patients in tqdm(train_patients):
        #sleep(0.01)
        print(patients[-14:])
        dcm_File = patients[-14:]
        roi_files = []
        mask_files = []

# Iterate over files in the directory
        
        try:
            for file_name in os.listdir(os.path.join(train_data_path,dcm_File)):
                if file_name.endswith("_roi.npy"):
                    roi_files.append(file_name)
                elif file_name.endswith("_label.npy"):
                    mask_files.append(file_name)
            roi_files.sort()
            mask_files.sort()

            for roi_file, mask_file in zip(roi_files, mask_files):
                print(os.path.join(train_data_path,dcm_File, roi_file))
                print(os.path.join(train_data_path,dcm_File, mask_file))
                roi_path = os.path.join(train_data_path,dcm_File, roi_file)
                img=np.load(roi_path)
                ROI = sitk.GetImageFromArray(img)
            
                mask_path = os.path.join(train_data_path,dcm_File, mask_file)
                msk=np.load(mask_path)
                Mask = sitk.GetImageFromArray(msk)
            
                features, feature_info = Extract_Features(ROI, Mask, 'params.yaml')
                
                #print(features)
                p_name=roi_file
                print(p_name)
                features['Patient_id']=p_name
                
                malignancy=  data.loc[data['ROI Image Name'] == p_name].Malignancy
                print(malignancy.any())
                features['Malignancy']=malignancy.any()
                print(features['Malignancy'])
                if malignancy.any() != None:
                    Feature.append(features)
        except Exception as Error:
            with open('Error_list_radiomics', 'a') as error_file:
                error_file.write(f"Error: {dcm_File}:{str(Error)}\n")
                print("Error: "+dcm_File+":"+str(Error)+"\n")
            continue
    
    df = DF(Feature).fillna('0')
    print(df.head(5))
    columns_to_move = ['Patient_id'	,	'Malignancy']
# List of columns that are not in the specified order
    remaining_columns = [col for col in df.columns if col not in columns_to_move]

# Create a new DataFrame with the desired column order
    new_order = columns_to_move + remaining_columns
    df = df[new_order]
    df['Malignancy'] = df['Malignancy'].map({True: 1, False: 0}).fillna(-1)
    df.to_csv('./Radiomics_Feature_Augmented.csv',index=False,sep=',')

    #end = time.clock()
    #print(end-start)  