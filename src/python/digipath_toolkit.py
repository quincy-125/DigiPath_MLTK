"""
digipath_toolkit.py

process large slide images using openslide
"""
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
                            parser utility: 
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

"""
                            notebook & development convenience
"""

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
                            patch wrangling functions
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


def get_sample_selection_mask(small_im, patch_select_method, run_parameters=None):
    """ Usage: mask_im = get_sample_selection_mask(small_im, patch_select_method)

    Args:
        small_im:               selection image
        patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'

    Returns:
        mask_im:                numpy boolean matrix size of small_im

    """
    if not run_parameters is None and 'rgb2lab_threshold' in run_parameters:
        rgb2lab_threshold = run_parameters['rgb2lab_threshold']
    else:
        rgb2lab_threshold = 80
    #                   initialize the return value
    mask_im = None

    if patch_select_method == 'threshold_rgb2lab':
        thresh = rgb2lab_threshold
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
    #                   initialize an empty return value
    patch_location_array = []
    #                   name the input variables
    wsi_filename = run_parameters['wsi_filename']
    thumbnail_divisor = run_parameters['thumbnail_divisor']
    patch_select_method = run_parameters['patch_select_method']
    patch_height = run_parameters['patch_height']
    patch_width = run_parameters['patch_width']
    #                   set defaults for newly added parameters
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

    #                   get the start, stop locations list for the rows
    pixels_height = obj_level_diminsions[image_level][1]
    rows_fence_array = get_fence_array(patch_length=patch_height, overall_length=pixels_height)

    #                   get the start, stop locations list for the columns
    pixels_width = obj_level_diminsions[image_level][0]
    cols_fence_array = get_fence_array(patch_length=patch_width, overall_length=pixels_width)

    #                   get a thumbnail image for the patch select method
    thumbnail_size = (pixels_width // thumbnail_divisor, pixels_height // thumbnail_divisor)
    small_im = os_im_obj.get_thumbnail(thumbnail_size)
    os_im_obj.close()
    #                     OpenSlide close                     #

    #                   get the binary mask as a measure of image region content
    mask_im = get_sample_selection_mask(small_im, patch_select_method)

    #                   iterator for rows:  (top_row, bottom_row, full_scale_row_number)
    it_rows = zip(rows_fence_array[:, 0] // thumbnail_divisor,
                  rows_fence_array[:, 1] // thumbnail_divisor,
                  rows_fence_array[:, 0])

    #                   variables for columns iterator
    lft_cols = cols_fence_array[:, 0] // thumbnail_divisor
    rgt_cols = cols_fence_array[:, 1] // thumbnail_divisor
    cols_array = cols_fence_array[:, 0]

    for tmb_row_top, tmb_row_bot, row_n in it_rows:
        #               iterator for cols:  (left_column, right_column, full_scale_column_number)
        it_cols = zip(lft_cols, rgt_cols, cols_array)

        for tmb_col_lft, tmb_col_rgt, col_n in it_cols:

            #           if the sum of the mask elements is larger than the threshold...
            if (mask_im[tmb_row_top:tmb_row_bot, tmb_col_lft:tmb_col_rgt]).sum() > threshold:

                #       add the full scale row and column of the upper left corner to the list
                patch_location_array.append((row_n, col_n))

    return patch_location_array

"""
                            visualization | examination
"""

def get_patch_locations_preview_imagefor_image_level(run_parameters):
    """ Usage:
    mask_image, thumb_preview, patch_location_array = get_patch_locations_preview_imagefor_image_level(run_parameters)

    create viewable images to show patch locations

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
    #                   unpack - name the variables
    wsi_filename = run_parameters['wsi_filename']
    patch_select_method = run_parameters['patch_select_method']
    thumbnail_divisor = run_parameters['thumbnail_divisor']
    border_color = run_parameters['border_color']

    #                   scale the patch size to the thumbnail image
    scaled_patch_height = run_parameters['patch_height'] // thumbnail_divisor - 1
    scaled_patch_width = run_parameters['patch_width'] // thumbnail_divisor - 1

    if 'image_level' in run_parameters:
        image_level = run_parameters['image_level']
    else:
        image_level = 0

    #                     OpenSlide open                      #
    os_im_obj = openslide.OpenSlide(wsi_filename)

    #                   get the size of the image at this image level
    obj_level_diminsions = os_im_obj.level_dimensions
    pixels_width = obj_level_diminsions[image_level][0]
    pixels_height = obj_level_diminsions[image_level][1]

    #                   get the thumbnail image scaled to the thumbnail divisor
    thumbnail_size = (pixels_width // thumbnail_divisor, pixels_height // thumbnail_divisor)
    thumb_preview = os_im_obj.get_thumbnail(thumbnail_size)
    os_im_obj.close()
    #                     OpenSlide close                      #

    #                   get the mask image for this patch_select_method
    mask_image = get_sample_selection_mask(thumb_preview, patch_select_method)
    #                   convert it from a binary matrix to a viewable image
    mask_image = tip.Image.fromarray(np.uint8(mask_image * 255), 'L')

    #                   rectangle-drawing object for the thumbnail preview image
    thumb_draw = ImageDraw.Draw(thumb_preview)

    #                   thumbnail-scaled patch locations to draw rectangles
    patch_location_array = get_patch_location_array_for_image_level(run_parameters)

    #                   draw the rectangles on the thumb_preview image
    for r, c in patch_location_array[:]:
        #               upper left corner = scaled column & row location of patch on thumbnail image
        ulc = (c // thumbnail_divisor, r // thumbnail_divisor)
        #               lower right corner = upper left corner + scaled patch size
        lrc = (ulc[0] + scaled_patch_width, ulc[1] + scaled_patch_height)
        
        #               draw the rectangle from the upper left corner to the lower right corner
        thumb_draw.rectangle((ulc, lrc), outline=border_color, fill=None)

    return mask_image, thumb_preview, patch_location_array


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

    #               get the two images and the location array
    mask_image, thumb_preview, patch_location_array = get_patch_locations_preview_imagefor_image_level(run_parameters)

    #               name and write the thumb_preview image
    thumb_preview_filename = os.path.join(output_dir, wsi_file_base + 'marked_thumb.jpg')
    with open(thumb_preview_filename, 'w') as fh:
        thumb_preview.save(fh)

    #               name and write the mask_preview image
    mask_preview_filename = os.path.join(output_dir, wsi_file_base + 'mask.jpg')
    with open(mask_preview_filename, 'w') as fh:
        mask_image.save(fh)

    #               name, build and write a dataframe for the upper left corners list
    location_array_filename = os.path.join(output_dir, wsi_file_base + 'patch_locations.tsv')
    patchlocation_df = pd.DataFrame(patch_location_array, columns=['row', 'col'])
    patchlocation_df.index.name = '#'
    patchlocation_df.to_csv(location_array_filename, sep='\t')

    #               print the output files location
    print('mask preview set saved:\n\t%s\n\t%s\n\t%s'%(thumb_preview_filename,
                                                       mask_preview_filename,
                                                       location_array_filename))

"""
                            image generation
"""

class PatchImageGenerator():
    """
    General case patch image generator for openslide Whole Slide Image file types

    Usage:  patch_image_generator = PatchImageGenerator(run_parameters)
            patch_dict = patch_image_generator.next_patch()

    Args:
        run_parameters:         (with these keys)
                                wsi_filename:           file name (with valid path)
                                patch_height:           patch size = (patch_width, patch_height)
                                patch_width:            patch size = (patch_width, patch_height)
                                thumbnail_divisor:      wsi_image full size divisor to create thumbnail image
                                patch_select_method:    'threshold_rgb2lab' or 'threshold_otsu'
                                threshold:              minimimum sum of thresholded image (default = 0)
    yields:
        patch_dict:             (with these keys)
                                patch_image:            PIL image of patch size
                                image_level_x:          column location in image level image
                                image_level_y:          row location in image level image
                                level_0_x:              column location in image (level 0)
                                level_0_y:              row location in image (level 0)
    """

    def __init__(self, run_parameters):
        self.os_obj = openslide.OpenSlide(run_parameters['wsi_filename'])
        self.image_level = run_parameters['image_level']
        self.patch_size = (run_parameters['patch_width'], run_parameters['patch_height'])
        _multi_ = self.os_obj.level_downsamples[self.image_level]
        self.image_level_loc_array = get_patch_location_array_for_image_level(run_parameters)
        self.level_0_location_array = [(int(p[0] * _multi_), int(p[1] * _multi_)) for p in self.image_level_loc_array]
        self._number_of_patches = len(self.image_level_loc_array)
        self._patch_number = -1

    def __del__(self):
        self.os_obj.close()

    def next_patch(self):
        self._patch_number += 1
        if self._patch_number < self._number_of_patches:
            patch_dict = {'patch_number': self._patch_number}
            patch_dict['image_level_x'] = self.image_level_loc_array[self._patch_number][1]
            patch_dict['image_level_y'] = self.image_level_loc_array[self._patch_number][0]
            patch_dict['level_0_x'] = self.level_0_location_array[self._patch_number][1]
            patch_dict['level_0_y'] = self.level_0_location_array[self._patch_number][0]

            location = (patch_dict['level_0_x'], patch_dict['level_0_y'])
            patch_dict['patch_image'] = self.os_obj.read_region(location, self.image_level, self.patch_size)

            return patch_dict

        else:
            raise StopIteration()


def image_file_to_patches_directory_for_image_level(run_parameters):
    """ Usage: image_file_to_patches_directory_for_image_level(run_parameters)

    Args:
        run_parameters:         (python dict with these keys)
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
                                (prints number of images written)
    """
    # explicitly name the input parameters
    image_file_name = run_parameters['wsi_filename']
    if os.path.isfile(image_file_name) == False:
        print('\n\n\tFile not found:\n\t%s\n\n'%(image_file_name))
        return
    output_dir = run_parameters['output_dir']
    class_label = run_parameters['class_label']

    if 'file_ext' in run_parameters:
        file_ext = run_parameters['file_ext']
        if file_ext[0] != '.':
            file_ext = '.' + file_ext
    else:
        file_ext = '.jpg'

    # check / ceate the output directory
    if os.path.isdir(output_dir) == False:
        print('creating output directory:\n%s\n' % (output_dir))
        os.makedirs(output_dir)

    # prepare name generation dictionary
    _, file_name_base = os.path.split(image_file_name)
    file_name_base, _ = os.path.splitext(file_name_base)
    patch_image_name_dict = {'case_id': file_name_base, 'class_label': class_label, 'file_ext': file_ext}

    # get the patch-image generator object
    patch_generator_obj = PatchImageGenerator(run_parameters)
    patch_number = -2
    while True:
        # iterate the patch_generator_obj untill empty
        try:
            patch_dict = patch_generator_obj.next_patch()
            patch_image_name_dict['location_x'] = patch_dict['image_level_x']
            patch_image_name_dict['location_y'] = patch_dict['image_level_y']
            patch_name = dict_to_patch_name(patch_image_name_dict)
            patch_full_name = os.path.join(output_dir, patch_name)
            # write the file
            patch_dict['patch_image'].convert('RGB').save(patch_full_name)
            patch_number = patch_dict['patch_number']

        except StopIteration:
            print('%i images written' % (patch_number + 1))
            break