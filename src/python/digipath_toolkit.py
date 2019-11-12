import os
from collections import OrderedDict
import argparse

import numpy as np
import pandas as pd
import yaml

from skimage.filters import threshold_otsu
from skimage.color import rgb2lab

from PIL import ImageDraw
from PIL import TiffImagePlugin as tip

import openslide

"""
            Utilities:
"""
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
    """ Usage: run_parameters = get_run_parameters(run_directory, run_file)
        Read the input arguments into a dictionary
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


def get_file_size_ordered_dict(data_dir, file_type_list):
    """ Usage:  file_size_ordered_dict = get_file_size_ordered_dict
        get size-ranked list of files of type in a directory

    Args:
        data_dir:           path to directory
        file_type_list:     file type extensions list (including period) e.g. = ['.svs', '.tif', '.tiff']

    Returns:
        ordered_dictionary: file_name: file_size   (ordered by file size)
    """
    file_size_ordered_dict = OrderedDict()
    if os.path.isdir(data_dir) == True:
        file_size_dict = {}
        for f in os.listdir(data_dir):
            ff = os.path.join(data_dir, f)
            if os.path.isfile(ff):
                _, f_ext = os.path.splitext(ff)
                if f_ext in file_type_list:
                    file_size_dict[f] = os.path.getsize(ff)

        file_size_dict_keys = list(file_size_dict.keys())
        sizes_idx = np.argsort(np.array(list(file_size_dict.values())))
        for idx in sizes_idx:
            k = file_size_dict_keys[idx]
            file_size_ordered_dict[k] = file_size_dict[k]
    else:
        file_size_ordered_dict['Directory Not Found: ' + data_dir] = 0

    return file_size_ordered_dict


def get_level_sizes_dict(image_file_name):
    """
    Usage:  level_sizes_dict = get_level_sizes_dict(image_file_name)
            read an openslide image type file to get the pyramid sizes available

    Args:
        image_file_name: full path or on path file name of .svs or some openslide format

    Returns:
        level_sizes_dict:
                            level_sizes_dict['image_size'] = os_obj.dimensions
                            level_sizes_dict['level_count'] = os_obj.level_count
                            level_sizes_dict["level_downsamples"] = os_obj.level_downsamples
                            level_sizes_dict['level_diminsions'] = os_obj.level_dimensions
    """
    level_sizes_dict = dict()
    os_obj = openslide.OpenSlide(image_file_name)
    level_sizes_dict['image_size'] = os_obj.dimensions
    level_sizes_dict['level_count'] = os_obj.level_count
    level_sizes_dict["level_downsamples"] = os_obj.level_downsamples
    level_sizes_dict['level_diminsions'] = os_obj.level_dimensions
    os_obj.close()

    return level_sizes_dict


def lineprint_level_sizes_dict(image_file_name):
    """
    Usage:  lineprint_level_sizes_dict(image_file_name)
            display the openslide image type file pyramid sizes available

    Args:
        image_file_name: full path or on path file name of .svs or some openslide format

    Returns:
        None:               (prints)
                            'image_size': os_obj.dimensions
                            'level_count': os_obj.level_count
                            'level_downsamples': os_obj.level_downsamples
                            'level_diminsions': os_obj.level_dimensions
    """
    level_sizes_dict = get_level_sizes_dict(image_file_name)
    key_list = ['image_size', 'level_count', 'level_diminsions', 'level_downsamples']
    print(' ')
    for k in key_list:
        if k in level_sizes_dict:
            print('%20s: ' % (k), level_sizes_dict[k])
        else:
            print('%20s: not found' % (k))
    print(' ')

"""
            functions needed by most methods
"""
def dict_to_patch_name(patch_image_name_dict):
    """ Usage: patch_name = dict_to_patch_name(patch_image_name_dict)
        convert the dictionary into a file name string
    
    Args:
        patch_image_name_dict:  {'case_id': 'd83cc7d1c94', 
                                 'location_x': 100, 
                                 'location_y': 340, 
                                 'class_label': 'dermis', 
                                 'file_type': '.jpg' }
        
    Returns:
        patch_name:     file name (without directory path)
    """
    if len(patch_image_name_dict['file_ext']) > 1 and patch_image_name_dict['file_ext'][0] != '.':
        patch_image_name_dict['file_ext'] = '.' + patch_image_name_dict['file_ext']
        
    patch_name = patch_image_name_dict['case_id']
    patch_name += '_%i'%patch_image_name_dict['location_x']
    patch_name += '_%i'%patch_image_name_dict['location_y'] 
    patch_name += '_%s'%patch_image_name_dict['class_label']
    patch_name += '%s'%patch_image_name_dict['file_ext']
    
    return patch_name


def patch_name_to_dict(patch_file_name):
    """ Usage: patch_image_name_dict = patch_name_to_dict(patch_file_name)
        convert a file name string into a dictionary

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
    """ Usage: fence_array = get_fence_array(patch_length, overall_length)
        create a left-right set of pairs that descrete overall_length into patch_length segments

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
    """ Usage: mask_im = get_sample_selection_mask(small_im, patch_select_method)

    Args:
        small_im:               selection image
        patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'

    Returns:
        mask_im:                numpy boolean matrix size of small_im

    """
    mask_im = None

    if patch_select_method == 'threshold_rgb2lab':
        thresh = 80
        np_img = np.array(small_im.convert('RGB'))
        np_img = rgb2lab(np_img)
        np_img = np_img[:, :, 0]
        mask_im = np.array(np_img) < thresh

    elif patch_select_method == 'threshold_otsu':
        grey_thumbnail = np.array(small_im.convert('L'))
        thresh = threshold_otsu(grey_thumbnail)
        mask_im = np.array(grey_thumbnail) < thresh

    else:
        print('patch_select_method %s not implemented' % (patch_select_method))

    return mask_im


def get_patch_location_array_for_image_level(run_parameters):
    """ Usage: patch_location_array = get_patch_location_array_for_image_level(run_parameters)
        using 'patch_select_method", find all upper left corner locations of patches
        that won't exceed image size givin the 'patch_height' and 'patch_width'

    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
                                threshold:              minimimum sum of thresholded image (default = 0)
                                image_level:            openslide image pyramid level 0,1,2,...
    Returns:
        patch_location_array

    """
    patch_location_array = []

    wsi_filename = run_parameters['wsi_filename']
    thumbnail_divisor = run_parameters['thumbnail_divisor']
    patch_select_method = run_parameters['patch_select_method']
    patch_height = run_parameters['patch_height']
    patch_width = run_parameters['patch_width']

    if 'threshold' in run_parameters:
        threshold = run_parameters['threshold']
    else:
        threshold = 0

    if 'image_level' in run_parameters:
        image_level = run_parameters['image_level']
    else:
        image_level = 0

    #                     OpenSlide open                      #
    os_im_obj = openslide.OpenSlide(wsi_filename)
    obj_level_diminsions = os_im_obj.level_dimensions

    pixels_height = obj_level_diminsions[image_level][1]
    rows_fence_array = get_fence_array(patch_length=patch_height, overall_length=pixels_height)

    pixels_width = obj_level_diminsions[image_level][0]
    cols_fence_array = get_fence_array(patch_length=patch_width, overall_length=pixels_width)

    thumbnail_size = (pixels_width // thumbnail_divisor, pixels_height // thumbnail_divisor)
    small_im = os_im_obj.get_thumbnail(thumbnail_size)
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
            if (mask_im[tmb_row_top:tmb_row_bot, tmb_col_lft:tmb_col_rgt]).sum() > threshold:
                patch_location_array.append((row_n, col_n))

    return patch_location_array


def get_patch_location_array(run_parameters):
    """ Usage: patch_location_array = get_patch_location_array(run_parameters)
        using 'patch_select_method", find all upper left corner locations of patches
        that won't exceed image size givin the 'patch_height' and 'patch_width'

    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
    Returns:
        patch_location_array

    """
    return get_patch_location_array_for_image_level(run_parameters)


def get_patch_locations_preview_imagefor_image_level(run_parameters):
    """ Usage:
    mask_image, thumb_preview, patch_location_array = get_patch_locations_preview_imagefor_image_level(run_parameters)
    get the images and data needed to display where the patches are for the input parameters

    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                border_color:           patch-box representation color 'red', 'blue' etc
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
                                threshold:              sum of thresholded image minimimum (default = 0)
                                image_level:            openslide image pyramid level 0,1,2,...
    Returns:
        mask_image:             black & white image of the mask
        thumb_preview:          thumbnail image with patch locations marked
        patch_location_array:   list of patch locations used [(row, col), (row, col),... ]

    """
    wsi_filename = run_parameters['wsi_filename']
    patch_select_method = run_parameters['patch_select_method']
    thumbnail_divisor = run_parameters['thumbnail_divisor']
    patch_height = run_parameters['patch_height'] // thumbnail_divisor - 1
    patch_width = run_parameters['patch_width'] // thumbnail_divisor - 1
    border_color = run_parameters['border_color']

    if 'image_level' in run_parameters:
        image_level = run_parameters['image_level']
    else:
        image_level = 0

    #                     OpenSlide open                      #
    os_im_obj = openslide.OpenSlide(wsi_filename)

    obj_level_diminsions = os_im_obj.level_dimensions

    pixels_width = obj_level_diminsions[image_level][0]
    pixels_height = obj_level_diminsions[image_level][1]

    thumbnail_size = (pixels_width // thumbnail_divisor, pixels_height // thumbnail_divisor)
    thumb_preview = os_im_obj.get_thumbnail(thumbnail_size)
    os_im_obj.close()

    mask_image = get_sample_selection_mask(thumb_preview, patch_select_method)
    mask_image = tip.Image.fromarray(np.uint8(mask_image * 255), 'L')

    thumb_draw = ImageDraw.Draw(thumb_preview)
    patch_location_array = get_patch_location_array_for_image_level(run_parameters)

    for r, c in patch_location_array[:]:
        ulc = (c // thumbnail_divisor, r // thumbnail_divisor)
        lrc = (ulc[0] + patch_width, ulc[1] + patch_height)
        thumb_draw.rectangle((ulc, lrc), outline=border_color, fill=None)

    return mask_image, thumb_preview, patch_location_array


def get_patch_locations_preview_image(run_parameters):
    """ Usage: mask_image, thumb_preview, patch_location_array = get_patch_locations_preview_image(run_parameters)
        get the images and data needed to display where the patches are for the input parameters

    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                border_color:           patch-box representation color 'red', 'blue' etc
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'

    Returns:
        mask_image:             black & white image of the mask
        thumb_preview:          thumbnail image with patch locations marked
        patch_location_array:   list of patch locations used [(row, col), (row, col),... ]

    """
    return get_patch_locations_preview_imagefor_image_level(run_parameters)


def write_mask_preview_set(run_parameters):
    """
    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                output_dir:             where the three files will be written
                                border_color:           patch-box representation color 'red', 'blue' etc
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
    Returns:
        None:               Writes three files:
                                wsi_basename_marked_thumb
                                wsi_basename_mask
                                wsi_basename_patch_locations
    """
    output_dir = run_parameters['output_dir']
    wsi_filename = run_parameters['wsi_filename']
    _, wsi_file_base = os.path.split(wsi_filename)
    wsi_file_base, _ = os.path.splitext(wsi_file_base)

    mask_image, thumb_preview, patch_location_array = get_patch_locations_preview_imagefor_image_level(run_parameters)

    thumb_preview_filename = os.path.join(output_dir, wsi_file_base + 'marked_thumb.jpg')
    with open(thumb_preview_filename, 'w') as fh:
        thumb_preview.save(fh)

    mask_preview_filename = os.path.join(output_dir, wsi_file_base + 'mask.jpg')
    with open(mask_preview_filename, 'w') as fh:
        mask_image.save(fh)

    location_array_filename = os.path.join(output_dir, wsi_file_base + 'patch_locations.tsv')
    patchlocation_df = pd.DataFrame(patch_location_array, columns=['row', 'col'])
    patchlocation_df.index.name = '#'
    patchlocation_df.to_csv(location_array_filename, sep='\t')

    print('mask preview set saved:\n\t%s\n\t%s\n\t%s'%(thumb_preview_filename,
                                                       mask_preview_filename,
                                                       location_array_filename))

"""
            Use Case implements
"""

def image_file_to_patches_directory_for_image_level(run_parameters):
    """ Usage: number_images_found = image_file_to_patches_directory_for_image_level(run_parameters)

    Args (run_parameters):  python dict.keys()
                                wsi_filename:           file name (with valid path)
                                output_dir:             writeable directory for the tfrecord
                                class_label:            label for all images
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                file_ext:               default is '.jpg' ('.png') was also tested
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
                                threshold:              minimimum sum of thresholded image (default = 0)

    Returns:                    None - writes images to output_dir (possibly many)
                                (prints number_images_found after all else)

    """
    image_file_name = run_parameters['wsi_filename']
    output_dir = run_parameters['output_dir']
    class_label = run_parameters['class_label']
    patch_width = run_parameters['patch_width']
    patch_height = run_parameters['patch_height']
    patch_size = (patch_width, patch_height)

    image_level = run_parameters['image_level']

    if 'file_ext' in run_parameters:
        file_ext = run_parameters['file_ext']
        if file_ext[0] != '.':
            file_ext = '.' + file_ext
    else:
        file_ext = '.jpg'

    if os.path.isdir(output_dir) == False:
        print('creating output directory:\n%s\n' % (output_dir))
        os.makedirs(output_dir)

    if len(file_ext) == 0:
        file_ext = '.jpg'
    elif file_ext[0] != '.':
        file_ext = '.' + file_ext

    _, file_name_base = os.path.split(image_file_name)
    file_name_base, _ = os.path.splitext(file_name_base)

    level_sizes_dict = get_level_sizes_dict(run_parameters['wsi_filename'])
    size_multiplier = level_sizes_dict['level_downsamples'][image_level]

    patch_location_array = get_patch_location_array_for_image_level(run_parameters)
    patch_location_array = [(int(p[0] * size_multiplier), int(p[1] * size_multiplier)) for p in patch_location_array]

    number_images_found = len(patch_location_array)
    patch_image_name_dict = {'case_id': file_name_base, 'class_label': class_label, 'file_ext': file_ext}

    # get the OpenSlide object - open the file, and get the mask with the scaled grids
    os_obj = openslide.OpenSlide(image_file_name)

    # iterate the list of locations found
    for read_location in patch_location_array:
        # build the patch name
        patch_image_name_dict['location_x'] = read_location[1]
        patch_image_name_dict['location_y'] = read_location[0]
        patch_name = dict_to_patch_name(patch_image_name_dict)
        patch_full_name = os.path.join(output_dir, patch_name)
        location = (read_location[1], read_location[0])

        # OpenSlide extract, convert, save
        patch_image = os_obj.read_region(location=location, level=image_level, size=patch_size)
        patch_image = patch_image.convert('RGB')
        patch_image.save(patch_full_name)

    os_obj.close()
    print('%i images found' % (number_images_found))
