

from lidcxmlparser import *
import pydicom as dicom
import numpy as np
import os
import glob
from skimage import draw, measure
import scipy
from collections import Counter
import csv
def most_common_value(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)

    if most_common:
        return most_common[0][0]
    else:
        return None

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(s) for s in glob.glob(path+'/*.dcm')]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def resample(img, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    image = img['array']
    spacing = img['Spacing']
    img_size = np.array(image.shape)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    image = scipy.ndimage.zoom(image, real_resize_factor)
    return image, img_size, real_resize_factor

def normalize_hu(image):
    MIN_BOUND = -1200.0
    MAX_BOUND = 600.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    image = (image*255).astype('uint8')
    return image

def crop_roi(resampled_img, img_size, seed_pos, crop_size, resize_factor):
    initial_seed = [seed_pos[0], seed_pos[1], seed_pos[2]]
    trans_seed = initial_seed*resize_factor
    start = []
    end= []
    for i in range(3):
        s = np.floor(trans_seed[i]-(crop_size[i]/2))
        e = np.ceil(trans_seed[i]+(crop_size[i]/2))
        if s<0:
            s = 0
        if e>resampled_img.shape[i]:
            e = resampled_img.shape[i]
        if e-s != crop_size[i]:
            pad = e-s-crop_size[i]
            if s==0:
                e = e-pad
            else:
                s = s+pad
        start.append(int(s))
        end.append(int(e))       
#    print(start,end,pad)
    roi = resampled_img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]      
        
    return roi
    
def get_nodule_center(xml_path, image, slice_loc):
    gt = LIDCXmlParser(xml_path)
    gt.parse()
    mask = np.zeros(image.shape)
    malignancy={}
    for indx, rad in enumerate(gt.rad_annotations): #has 4 radiologistes
        mask_1 = np.zeros(image.shape)
        m=[]
        print(rad.id)
        for nod in rad.nodules: #nod is one NormalNodule
            dist_m={}
            dist_m[nod.id]=nod.characterstics.malignancy
            
            for nod_roi in nod.rois:  
                #print(nod_roi)
                z_index = np.where(slice_loc==nod_roi.z)[0][0]
                xy = np.array(nod_roi.roi_xy)
                xx, yy = draw.polygon(xy[:,1],xy[:,0])
                mask_1[xx,yy,z_index] = 1
            
            m.append(dist_m)
        mask = mask+mask_1
        malignancy[rad.id]=m
        
    mask = np.array(mask>1).astype(int)
    L_mask = measure.label(mask)
    L_props= measure.regionprops(L_mask)
    center_pos = []
    for props in L_props:
        
        center = np.array(props.centroid).astype(int)
        center_pos.append(center)
    print(len(center_pos),(mask.shape))
    return center_pos,mask

  
def search_xml(file_dir):  
    xml_path=[]  
    for root, dirs, files in os.walk(file_dir): 
        for file in files: 
            if os.path.splitext(file)[1] == '.xml': 
                xml_path.append(os.path.join(root, file)) 
    return xml_path

if __name__ == '__main__':
    
    home_path = 'D:\Mtech Project\Dataset'
    save_path = './Nodule_crop'
    error_file="Error_list_1.txt"
    csv_file_path = 'example_1_1.csv'
    header = ['Patient_ID', 'Nodule No','Image File', 'Mask File','Malignancy']
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


    data_list = os.listdir(home_path)
    
    for patient_path in data_list:
        try:
            img_path = os.path.join(home_path,patient_path)
            xml_listpath = search_xml(img_path)
            for xml_path in xml_listpath:
                path, xml_f = os.path.split(xml_path)
                dcm = [s for s in glob.glob(path+'/*.dcm')]
                if len(dcm) > 10:
                    dicom_slices = load_scan(path)
                    image = [s.pixel_array*int(s.RescaleSlope)+int(s.RescaleIntercept) for s in dicom_slices ]
                    image = np.array(image).transpose(1,2,0)
                    slice_loc = np.array([s.ImagePositionPatient[2] for s in dicom_slices]).astype(float)
                    spacing = np.array([dicom_slices[0].PixelSpacing[0], 
                                        dicom_slices[0].PixelSpacing[1],
                                        dicom_slices[0].SliceThickness]).astype(float)
                    center_pos,mask = get_nodule_center(xml_path, image, slice_loc)
                    
                    if len(center_pos) != 0:
                        image_new = {}
                        image_new['array'] = image
                        image_new['Spacing'] = spacing
                        img, img_size, resize_factor = resample(image_new)
                        mask_new = {}
                        mask_new['array'] = mask
                        mask_new['Spacing'] = spacing
                        label, label_size, resize_factor = resample(mask_new)
                        img = normalize_hu(img)   
                        Nodule_num=0
                        print(len(center_pos))
                        for pos in center_pos:
                            Nodule_num = Nodule_num+1
                            seed_pos = [pos[0], pos[1], pos[2]]
                            ROI = crop_roi(img, img_size, seed_pos, [64,64,64] , resize_factor)
                            ROI_label = crop_roi(label, label_size, seed_pos, [64, 64, 64], resize_factor)
                            ROI = (ROI.astype(np.float32)-128)/128
                            if not os.path.exists(save_path+"/"+patient_path):
                                # Directory does not exist, so create it
                                os.makedirs(save_path+"/"+patient_path)
    #                       
                            roi_path=os.path.join(save_path+"/"+patient_path,patient_path+"_"+str(Nodule_num)+'_roi.npy')
                            mask_path=os.path.join(save_path+"/"+patient_path,patient_path+"_"+str(Nodule_num)+'_label.npy')
                            print(roi_path)
                            np.save(roi_path, ROI)
                            np.save(mask_path, ROI_label)
                            data=[patient_path,Nodule_num,roi_path,mask_path,None]
                            with open(csv_file_path, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(data)
                                
        except Exception as Error:
            with open('Error_list_1.txt', 'a') as error_file:
                error_file.write(f"Error: {patient_path}:{str(Error)}\n")
            continue
                        
        


        
    
