import os
# import sys
import argparse

import numpy as np
import yaml

from skimage.filters import threshold_otsu
from skimage.color import rgb2lab

import PIL.Image

def get_run_directory_and_run_file(args):
    """ Parse the input arguments to get the run_directory and run_file
    Args:
        system args:     -run_directory, -run_file (as below)

    Returns:
        run_directory:      where run_file is expected
        run_file:           yaml file with run parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    parser.add_argument('-run_file', type=str)
    args = parser.parse_args()

    run_directory = args.run_directory
    run_file = args.run_file

    return run_directory, run_file


def get_run_parameters(run_directory, run_file):
    """ Read the input arguments into a dictionary
    Args:
        run_directory:      where run_file is expected
        run_file:           yaml file with run parameters

    Returns:
        run_parameters:     python dictionary of run parameters
    """
    run_file_name = os.path.join(run_directory, run_file)
    with open(run_file_name, 'r') as fh:
        run_parameters = yaml.safe_load(fh)
    run_parameters['run_directory'] = run_directory
    run_parameters['run_file'] = run_file

    return run_parameters



def dict_to_patch_name(patch_image_name_dict):
    """ Usage:
    patch_name = dict_to_patch_name(patch_image_name_dict) 
    
    Args:
        patch_image_name_dict:  {'case_id': 'd83cc7d1c94', 
                                 'location_x': 100, 
                                 'location_y': 340, 
                                 'class_label': 'dermis', 
                                 'file_type': '.jpg' }
        
    Returns:
        patch_name:     file name (without directory path)
    """
    if patch_image_name_dict['file_ext'][0] != '.':
        patch_image_name_dict['file_ext'] = '.' + patch_image_name_dict['file_ext']
        
    patch_name = patch_image_name_dict['case_id']
    patch_name += '_%i'%patch_image_name_dict['location_x']
    patch_name += '_%i'%patch_image_name_dict['location_y'] 
    patch_name += '_%s'%patch_image_name_dict['class_label']
    patch_name += '%s'%patch_image_name_dict['file_ext']
    
    return patch_name


def patch_name_to_dict(patch_file_name):
    """ Usage:
    patch_image_name_dict = patch_name_to_dict(patch_file_name)
    
    Args:
        fname:          file name as created by get_patch_name()
        
    Returns:
        patch_image_name_dict:  {'case_id': field[0], 
                                 'location_x': int(field[1]), 
                                 'location_y': int(field[2]), 
                                 'class_label': field[3], 
                                 'file_ext': '.' + field[4] }
    """
    name_part, file_ext = os.path.splitext(patch_file_name)
    if len(file_ext) > 0 and file_ext[0] == '.':
        file_ext = file_ext[1:]

    name_field_list = name_part.split('_')
    
    patch_image_name_dict = {'case_id': name_field_list[0], 
                             'location_x': int(name_field_list[1]), 
                             'location_y': int(name_field_list[2]), 
                             'class_label': name_field_list[3], 
                             'file_ext': '.' + file_ext}
    
    return patch_image_name_dict


def get_fence_array(patch_length, overall_length):
    """ create a left-right set of pairs that descrete overall_length into patch_length segments
    Usage: fence_array = get_fence_array(patch_length, overall_length)

    Args:
        patch_length:   patch size - number of pixels high or wide
        patch_length:   overall number of pixels high or wide

    Returns:
        fence_array:    boundry values for each segment
        -----------:    [[left_0, right_0],
                         [left_1, right_1],
                         [left_2, right_2],... ]
    """
    # Determine the array size
    n_fenced = overall_length // patch_length  # number of boxes
    n_remain = 1 + overall_length % patch_length  # number of pixels leftover
    paddit = n_remain // 2  # padding for the beginning

    if n_remain == patch_length:  # exact fit special case: exactly one left over
        paddit = 0
        n_fenced = n_fenced + 1

    # Allocate as integers for use as indices
    fence_array = np.zeros((n_fenced, 2)).astype(int)
    for k in range(n_fenced):
        # for each box edge, get the beginning and end pixel location
        if k == 0:
            # first case special (padding)
            fence_array[k, 0] = paddit
            # add the width to it
            fence_array[k, 1] = fence_array[k, 0] + patch_length - 1

        elif fence_array[k - 1, 1] + patch_length <= overall_length:
            # Previous right pixel plus one
            fence_array[k, 0] = fence_array[k - 1, 1] + 1
            # add the width to it
            fence_array[k, 1] = fence_array[k, 0] + patch_length - 1

    return fence_array


def get_sample_selection_mask(small_im, patch_select_method):
    """ get an image mask  -  Under  development: unit test
    """
    mask_im = None

    if patch_select_method == 'threshold_rgb2lab':
        thresh = 80
        np_img = np.array(small_im.convert('RGB'))
        np_img = rgb2lab(np_img)
        np_img = np_img[:, :, 0]
        mask_im = np.array(np_img) < thresh
        # mask_im = PIL.Image.fromarray(np.uint8(mask_im) * 255)

    elif patch_select_method == 'threshold_otsu':
        grey_thumbnail = np.array(small_im.convert('L'))
        thresh = threshold_otsu(grey_thumbnail)
        mask_im = np.array(grey_thumbnail) < thresh
        # mask_im = PIL.Image.fromarray(np.uint8(mask_im) * 255)

    else:
        print('patch_select_method %s not implemented' % (patch_select_method))

    return mask_im


def get_patch_location_array(run_parameters):
    """ Usage: patch_location_array = get_patch_location_array(run_parameters)
    Args:
        run_parameters:     keys:
                            image_file_name,
                            thumbnail_divisor,
                            patch_select_method,
                            patch_height,
                            patch_width
    Returns:
        patch_location_array

    """
    patch_location_array = []

    image_file_name = run_parameters['image_file_name']
    thumbnail_divisor = run_parameters['thumbnail_divisor']
    patch_select_method = run_parameters['patch_select_method']
    patch_height = run_parameters['patch_height']
    patch_width = run_parameters['patch_width']

    #                     OpenSlide open                      #
    os_im_obj = openslide.OpenSlide(image_file_name)

    pixels_height = os_im_obj.dimensions[1]
    rows_fence_array = get_fence_array(patch_length=patch_height, overall_length=pixels_height)

    pixels_width = os_im_obj.dimensions[0]
    cols_fence_array = get_fence_array(patch_length=patch_width, overall_length=pixels_width)

    small_im = os_im_obj.get_thumbnail((pixels_height // thumbnail_divisor,
                                        pixels_width // thumbnail_divisor))
    os_im_obj.close()
    #                     OpenSlide close                     #

    mask_im = get_sample_selection_mask(small_im, patch_select_method)

    it_rows = zip(rows_fence_array[:, 0] // thumbnail_divisor,
                  rows_fence_array[:, 1] // thumbnail_divisor,
                  rows_fence_array[:, 0])

    lft_cols = cols_fence_array[:, 0] // thumbnail_divisor
    rgt_cols = cols_fence_array[:, 1] // thumbnail_divisor
    cols_array = cols_fence_array[:, 0]

    for tmb_row_top, tmb_row_bot, row_n in it_rows:
        it_cols = zip(lft_cols, rgt_cols, cols_array)
        for tmb_col_lft, tmb_col_rgt, col_n in it_cols:
            if (mask_im[tmb_row_top:tmb_row_bot, tmb_col_lft:tmb_col_rgt]).sum() > 0:
                patch_location_array.append((row_n, col_n))

    return patch_location_array
