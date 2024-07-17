
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom as pydi
import pylidc as pl
import glob
import pandas as pd
from skimage import measure
import os
import os


# %%
def get_cube_from_img(img3d, center, block_size):
    """"Code for this function is based on code from this repository: https://github.com/junqiangchen/LUNA16-Lung-Nodule-Analysis-2016-Challenge"""
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range

    center_x = center[0]
    center_y = center[1]
    center_z = center[2]

    block_size_x = block_size[0]
    block_size_y = block_size[1]
    block_size_z = block_size[2]

    start_x = max(center_x - block_size_x / 2, 0)
    if start_x + block_size_x > img3d.shape[0]:
        start_x = img3d.shape[0] - block_size_x

    start_y = max(center_y - block_size_y / 2, 0)
    if start_y + block_size_y > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size_y

    start_z = max(center_z - block_size_z / 2, 0)
    if start_z + block_size_z > img3d.shape[2]:
        start_z = img3d.shape[2] - block_size_z

    start_x = int(start_x)
    start_y = int(start_y)
    start_z = int(start_z)
    roi_img3d = img3d[start_x:start_x + block_size_x,
                start_y:start_y + block_size_y,
                start_z:start_z + block_size_z]
    return roi_img3d


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = np.array(image > 0.078, dtype=np.int8) + 1
    labels = measure.label(binary_image)
    background_label = labels[0, 0, 0]
    binary_image[background_label == labels] = 2
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1 - binary_image
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image
def get_slice(patient):
    s=0
    for subdir, dirs, files in os.walk(patient):
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))

        if len(dcms) >1:
            slices = [pydi.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices

def get_hu_values(image,slices):
    image = image.astype(np.int32)
    intercept = slices[0].RescaleIntercept  # transformation from pixels in their stored on disk representation to their in memory representation. U = m*SV + b,m=slope, sv=stored value, b=intercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    image = np.array(image, dtype=np.int32)
    image[image < -1000] = -1000
    image[image > 2000] = 2000
    #image = normalize(image)
    return image



data_path = "./Dataset/LIDC-IDRI*"
print(data_path)
data_list = sorted(glob.glob(data_path))
data_num = len(data_list)
print(data_num)
nodule_info = []

block_size = [64, 64, 64]

save_path_p = 'Data_new/classification/npyfiles/'

k = 1

print('------------------------------------------------')
error_list = []
for d_idx in range(data_num):

    pid = data_list[d_idx].split('/')[-1][-14:]

    print('processing-----{}'.format(pid))
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    print(scan)
    try:
        vol = scan.to_volume()

        print('------------------------------------------------')
        nods = scan.cluster_annotations()
        num_nods = len(nods)

        sid = scan.series_instance_uid

        pixel_info = scan.spacings  # x,y,z
        error_item = []

        for i, nod_i in enumerate(nods):

            num_name = str(k)
            if k < 10:
                num_name = '000' + num_name
            elif k < 100:
                num_name = '00' + num_name
            elif k < 1000:
                num_name = '0' + num_name

            cent = []
            diameter = 0
            mal_factor = 0
            bbox = []
            num_ann = len(nod_i)
            for j, ann_i in enumerate(nod_i):
                cent.append(ann_i.centroid)
                diameter += ann_i.diameter
                mal_factor += ann_i.feature_vals()[-1]
                bbox.append(ann_i.bbox_dims())
            cent = np.mean(cent, axis=0)
            diameter = diameter / num_ann
            bbox = np.max(bbox, axis=0)
            mal_factor = mal_factor / num_ann

            if mal_factor > 3:
                mal = 1.
            else:
                mal = 0.

            save_image_name = 'Image_{}_{}_{}_ROI.npy'.format(pid, num_name, mal)
            save_mask_name = 'Mask_{}_{}_{}_mask.npy'.format(pid, num_name, mal)
            patch = get_cube_from_img(vol, cent, block_size)
            slices=get_slice(data_list[d_idx])
            image=get_hu_values(patch,slices)
            mask = segment_lung_mask(patch, False)
            if patch.shape[0] < block_size[0]:
                error_item.append(save_image_name)
                print('error: index {}'.format(save_image_name))
            else:
                nodule_info.append([pid, sid, save_image_name, *pixel_info, *cent, *bbox, diameter, mal_factor, mal])
                if mal > 0:
                    save_path_f = save_path_p + '/True/'
                else:
                    save_path_f = save_path_p + '/False/'
                os.makedirs(save_path_f, exist_ok=True)
                np.save(save_path_f + save_image_name, patch)
                np.save(save_path_f + save_mask_name, mask)
            k += 1
    except Exception as rerror:
        error_list.append(scan.patient_id + ":" + str(rerror))
        print(scan.patient_id + ":" + str(rerror))
        continue

# %%
column_index = ['patient_id', 'serisuid', 'Ny file Name',
                'pixel_x', 'pixel_y', 'pixel_z',
                'interp_cent_x', 'interp_cent_y', 'interp_cent_z',
                'bbox_x', 'bbox_y', 'bbox_z',
                'diameter', 'malignancy_level', 'malignancy'
                ]
print(error_list)
nodule_info_csv = pd.DataFrame(np.array(nodule_info), columns=column_index)
nodule_info_csv = nodule_info_csv.set_index(column_index[0])
nodule_info_csv.to_csv('nodule_info_new.csv')
# nodule_info = pd.read_csv('nodule_info.csv')
