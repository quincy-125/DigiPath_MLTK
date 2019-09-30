"""
    module for converting OpenSlide compatable image files to TFRecord files 
    and viewing the record as a thumbnail
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tempfile import TemporaryDirectory
import argparse
import yaml

from PIL import ImageDraw

import IPython.display as ip_display
import os
import sys
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import skimage
from skimage.filters import threshold_otsu

import openslide

import PIL
from PIL.Image import Image


WORKING_THUMB_MAX_SIZE = [2048, 2048]
WORKING_THUMB_MIN_SIZE = [1024, 1024]


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

def run_imfile_to_tfrecord(run_parameters):
    """ svs_file_to_patches_tfrecord(svs_file_name, output_dir, patch_size, drop_threshold, file_ext=None) """
    print('call: %s\nwith run_parameters:\n'%(
        'svs_file_to_patches_tfrecord(svs_file_name, output_dir, patch_size, drop_threshold, file_ext=None)'))
    for k, v in run_parameters.items():
        print('%40s: %s'%(k,v))

def write_tfrecord_masked_thumbnail(run_parameters):
    """ get_tfrecord_masked_thumbnail(tfrecord_filename, wsi_filename, thumb_scale, alpha, border_color='blue') """
    print('call: %s\nwith run_parameters:\n'%(
        'get_tfrecord_masked_thumbnail(tfrecord_filename, wsi_filename, thumb_scale, alpha, border_color="blue")'))
    for k, v in run_parameters.items():
        print('%40s: %s'%(k,v))


def run_imfile_to_tfrecord(run_parameters):
    """ svs_file_to_patches_tfrecord(svs_file_name, output_dir, patch_size, drop_threshold, file_ext=None)
    """
    for k, v in run_parameters.items():
        print('%40s: %s'%(k, v))

def write_tfrecord_masked_thumbnail(run_parameters):
    """ get_tfrecord_masked_thumbnail(tfrecord_filename, wsi_filename, thumb_scale, alpha, border_color='blue') """
    for k, v in run_parameters.items():
        print('%40s: %s'%(k, v))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#                                                                                NEW Start    <o><o>
def image_patch(image_string, label, ulc_row, ulc_col, lrc_row, lrc_col, image_name='patch'):
    """ image_metadat_dict = image_example(image_string, label, image_name)
    Create a dictionary of jpg image features
    Args:
        image_string:  bytes(PIL_image)
        label:         a number
        image_name:    bytes(image_name)
    Returns:
        one_tf_train_example: tf.train.Example 
    """
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {'height': _int64_feature(image_shape[0]),
               'width': _int64_feature(image_shape[1]),
               'depth': _int64_feature(image_shape[2]),
               'ulc_row': _int64_feature(ulc_row),
               'ulc_col': _int64_feature(ulc_col),
               'lrc_row': _int64_feature(lrc_row),
               'lrc_col': _int64_feature(lrc_col),
               'label': _int64_feature(label), 
               'image_name': _bytes_feature(image_name),
               'image_raw': _bytes_feature(image_string) }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_image_patch_function(example_proto):
    """ reader for image_example() encoded as tfrecord file 
        usage:
    parsed_image_dataset = tf.data.TFRecordDataset(tfrecord_name).map(_parse_image_function)
    
    Args: 
        example_proto:
        
    Returns:
        iterable_tfrecord:   try iterable_tfrecord.__iter__()
    """
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'ulc_row': tf.io.FixedLenFeature([], tf.int64),
        'ulc_col': tf.io.FixedLenFeature([], tf.int64),
        'lrc_row': tf.io.FixedLenFeature([], tf.int64),
        'lrc_col': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string) }

    return tf.io.parse_single_example(example_proto, image_feature_description)


def get_iterable_tfrecord(tfr_name):
    """ usage:
    iterable_tfrecord = get_iterable_tfrecord(tfr_name)
    
    Args:
        tfr_name:   tensorflow data TFRecord file
        
    Returns:
        TFRecordDataset_map_object:  tf.data.TFRecordDataset().map()
        
    """
    return tf.data.TFRecordDataset(tfr_name).map(_parse_image_patch_function)
#                                                                                NEW End      <o><o>


def get_adjcent_segmented_length_fence_array(segment_length, length):
    """ Usage - still developing:
    fences = get_adjcent_segmented_length_fence_array(segment_length, length)
    
    Args:
        segment_length: patch size - number of pixels high or wide 
        length:         overall number of pixels high or wide 
        
    Returns:
        fences:         n x 2 array for [top, bottom] or [height, width] usage
                        Note that images are numbered from top to bottom
    """
    # allocate the return container 
    fences = {}
    n_fenced = length // segment_length           # overall length divided by patch size  >> number of boxes
    n_remain = 1 + length % segment_length        # remaining number of pixels
    paddit = n_remain // 2                        # padding for the beginning and end of the boxed section
    if n_remain == segment_length:                # if exactly one box is left over special case
        paddit = 0
        n_remain = 0
        n_fenced = n_fenced + 1
    
    # allocate the array
    fence_array = np.zeros((n_fenced, 2)).astype(int)
    for k in range(n_fenced):                     # for each box edge, get the beginning and end pixel location
        if k == 0:                                # first case special: start with padding
            fence_array[k, 0] = paddit
            fence_array[k, 1] = fence_array[k, 0] + segment_length - 1
        elif fence_array[k-1, 1] + segment_length <= length:
                                                  # Use the previous right pixel plus one 
            fence_array[k, 0] = fence_array[k-1, 1] + 1
            fence_array[k, 1] = fence_array[k, 0] + segment_length - 1
                                                  # add one width to the left pixel in this segment
                
    # return everything for development debug    
    fences['fence_array'] = fence_array
    fences['n_fenced'] = n_fenced
    fences['n_remain'] = n_remain
    fences['padding'] = paddit

    return fences

        
def get_patch_name_from_row_col(row, col, base_name='patch', file_ext='.jpg'):
    """ Usage:
    patch_name = get_patch_name_from_row_col(row,col,base_name='patch',file_ext='.jpg') 
    
    Args:
        row, col:       integer list e.g row = [0, 20]
        base_name:      beginning of the file name
        file_ext:       default is '.jpg' Note: include the period before the name
        
    Returns:
        patch_name:     file name (without directory path)
    """
    if file_ext[0] != '.':
        file_ext = '.' + file_ext
    patch_name = base_name + '_row_%i_%i'%(row[0], row[1])
    patch_name += '_col_%i_%i%s'%(col[0], col[1], file_ext)
    
    return patch_name


def get_row_col_from_patch_name(fname):
    """ Usage:
    row_col_dict = get_row_col_from_filename(fname) 
    
    Args:
        fname:          file name as created by this module function:
                        get_patch_name_from_row_col(row, col, base_name, file_ext)
    Returns:
        row_col_dict: { 'base_name': parts_list[0], 
                        'file_ext': file_ext, 
                        'row': row, 
                        'col': col }
    """
    row_label = 'row'
    col_label = 'col'
    r = []
    c = []
    base_name, file_ext = os.path.splitext(os.path.split(fname)[1])
    parts_list = base_name.split('_')
    
    for i in range(len(parts_list)):
        if parts_list[i] == row_label:
            r.append(parts_list[i+1])
            r.append(parts_list[i+2])
        elif parts_list[i] == col_label:
            c.append(parts_list[i+1])
            c.append(parts_list[i+2])
    row = np.array(r).astype(np.int)
    col = np.array(c).astype(np.int)
    
    return {'base_name': parts_list[0], 'file_ext': file_ext, 'row': row, 'col': col }


def get_mask_w_scale_grid(os_obj, patch_height, patch_width, thumbnail_divisor=None):
    """ Usage:
    mask_dict = get_mask_w_scale_grid(svs_file_name, patch_height, patch_width, thumbnail_divisor=None)
    
    Args:
        os_obj:         file name or opened OpenSlide object
        patch_height:   how high to make the patch indices
        patch_width:    how wide to make the patch indices
    
    Returns:
        mask_dict:      {'thumb_mask': mask_im, 
                         'full_scale_rows_dict': full_scale_rows_dict, 
                         'full_scale_cols_dict': full_scale_cols_dict, 
                         'full_scale_rows_arrays': full_scale_rows_arrays, 
                         'full_scale_cols_arrays': full_scale_cols_arrays,
                         'thumb_scale_rows_arrays': thumb_scale_rows_arrays, 
                         'thumb_scale_cols_arrays': thumb_scale_cols_arrays }
    """
    # don't close if it was passed in, close if it was opened in this function
    close_os_obj = False
    if isinstance(os_obj, str) and os.path.isfile(os_obj):
        os_obj = openslide.OpenSlide(os_obj)
        close_os_obj = True
    
    #                                                                                NEW fix start    <o><o>
    #                               get the indexing arrays for the full size grid
    pixels_height = os_obj.dimensions[1]
    pixels_width = os_obj.dimensions[0]

    full_scale_rows_dict = get_adjcent_segmented_length_fence_array(segment_length=patch_height, 
                                                                    length=pixels_height)
    full_scale_cols_dict = get_adjcent_segmented_length_fence_array(segment_length=patch_width, 
                                                                    length=pixels_width)
    full_scale_rows_arrays = full_scale_rows_dict['fence_array']
    full_scale_cols_arrays = full_scale_cols_dict['fence_array']
    
    #                               determine thumbnail size & get the mask
    pixels_height_ds = os_obj.level_dimensions[-1][1]
    pixels_width_ds = os_obj.level_dimensions[-1][0]
    #                                                                                NEW fix end       <o><o>
    
    if thumbnail_divisor is None:
        thumbnail_divisor = 1
        while_stopper = 20
        count = 0

        scale_determinant = max(pixels_height_ds, pixels_width_ds)
        while scale_determinant // thumbnail_divisor > WORKING_THUMB_MAX_SIZE[0] and count < while_stopper:
            count += 0
            thumbnail_divisor *= 2
    
    thumb_height = pixels_height_ds // thumbnail_divisor
    thumb_width = pixels_width_ds // thumbnail_divisor
    
    #                               get the indexing arrays for the thumbnail sized grid
    rows_divisor = pixels_height / thumb_height
    thumb_scale_rows_arrays = (full_scale_rows_arrays // rows_divisor).astype(int)
    cols_divisor = pixels_width / thumb_width
    thumb_scale_cols_arrays = (full_scale_cols_arrays // cols_divisor).astype(int)

    #                               git the mask image
    one_thumb = os_obj.get_thumbnail((thumb_height, thumb_width))
    grey_thumbnail = np.array(one_thumb.convert('L'))
    thresh = threshold_otsu(grey_thumbnail)
    mask = np.array(grey_thumbnail) < thresh
    mask_im = PIL.Image.fromarray(np.uint8(mask) * 255)
    
    # close if it was opened in this function - don't close if it was passed in
    if close_os_obj == True:
        os_obj.close()
        
    mask_dict = {'one_thumb': one_thumb,
                 'thumb_mask': mask_im, 
                 'thumbnail_divisor': thumbnail_divisor,
                 'full_scale_rows_dict': full_scale_rows_dict, 
                 'full_scale_cols_dict': full_scale_cols_dict, 
                 'full_scale_rows_arrays': full_scale_rows_arrays, 
                 'full_scale_cols_arrays': full_scale_cols_arrays,
                 'thumb_scale_rows_arrays': thumb_scale_rows_arrays, 
                 'thumb_scale_cols_arrays': thumb_scale_cols_arrays}
    
    return mask_dict


def svs_file_to_patches_tfrecord(svs_file_name, output_dir, patch_size, drop_threshold, file_ext=None):
    """ Usage:
    report_dict = svs_file_to_patches_tfrecord(svs_file_name, output_dir, patch_size, drop_threshold, file_ext)
    
    Args:
        svs_file_name:   accessable path file name
        output_dir:      writeable directory for the tfrecord
        patch_size:      list of 2 integers: [224, 224] or an integer if square
        drop_threshold:  number between 0 & 1 -- if the masked area of the patch is smaller it is included
        file_ext:        default is '.jpg' ('.png') was tested (Note the period is included)
        
    Returns:
        svs_file_conversion_dict:  {'mask_dict': mask_dict, 
                                    'tfrecord_file_name': tfrecord_file_name, 
                                    'number_of_patches': seq_number, 
                                    'temp_dir': temp_dir }
    """
    
    # construct file nameing variables from svs_file_name, output_dir, set file_ext if missing
    _, file_name_base = os.path.split(svs_file_name)
    file_name_base, _ = os.path.splitext(file_name_base)
    tfrecord_file_name = file_name_base + '.tfrecords'
    tfrecord_file_name = os.path.join(output_dir, tfrecord_file_name)
    if file_ext is None:
        file_ext = '.jpg'
        
    # expand patch size into both height and width
    if isinstance(patch_size, list) and len(patch_size) == 2:
        patch_height = patch_size[0]
        patch_width = patch_size[1]
    else:
        patch_height = patch_width = patch_size
    
    # get the OpenSlide object - open the file, and get the mask with the scaled grids
    os_obj = openslide.OpenSlide(svs_file_name)
    mask_dict = get_mask_w_scale_grid(os_obj, patch_height, patch_width)
    
    # Break-out option
    # mask_dict = get_mask_w_scale_grid(os_obj, patch_height, patch_width)
    
    # convert the dictionary to named variables for clarity
    mask_im = mask_dict['thumb_mask']
    full_scale_rows_arrays = mask_dict['full_scale_rows_arrays']
    full_scale_cols_arrays = mask_dict['full_scale_cols_arrays']
    thumb_scale_rows_arrays = mask_dict['thumb_scale_rows_arrays']
    thumb_scale_cols_arrays = mask_dict['thumb_scale_cols_arrays']
    
    seq_number = 0
    # open a temporary directory and a TFRecordWriter object
    with TemporaryDirectory() as temp_dir:
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            
            # iterate through the rows and columns of patches
            for row in range(full_scale_rows_arrays.shape[0]):
                for col in range(full_scale_cols_arrays.shape[0]):
                    # get the small segment of the mask with PIL ordering for crop
                    r = thumb_scale_rows_arrays[row]
                    c = thumb_scale_cols_arrays[col]
                    area = (c[0], r[0], c[1], r[1])
                    thumb_segment = mask_im.crop(area)
                    thumb_arr = np.array(thumb_segment)
                    
                    # Break-out option
                    # mask_dict = get_mask_w_scale_grid(os_obj, patch_height, patch_width)
                    
                    # evaluate the thumb_segment - reject if the masked (dark area) is too large
                    mask_value = np.float(np.sum(thumb_arr==0)) / np.float(np.prod(thumb_arr.shape))
                    if mask_value <= drop_threshold:
                        # put row=(top, bottom) & column=(left, right) fence arrays into PIL format
                        fs_row = full_scale_rows_arrays[row]
                        fs_col = full_scale_cols_arrays[col]
                        read_location = (full_scale_cols_arrays[col][0], full_scale_rows_arrays[row][0])
                        
                        # build the full-path patch name in the temp dir
                        patch_name = get_patch_name_from_row_col(fs_row, fs_col, file_name_base, file_ext)
                        patch_full_name = os.path.join(temp_dir, patch_name)

                        # let OpenSlide extract the (PIL) patch, convert to RGB & save temporary
                        full_sect = os_obj.read_region(level=0, size=patch_size, location=read_location)
                        full_sect = full_sect.convert('RGB')                                                
                        full_sect.save(patch_full_name)
                        
                        # add the file to the tfrecord
                        image_string = open(patch_full_name, 'rb').read()
                        tf_example_obj = image_patch(image_string, 
                                                     label=seq_number, 
                                                     ulc_row=r[0], 
                                                     ulc_col=c[0], 
                                                     lrc_row=r[1], 
                                                     lrc_col=c[1],
                                                     image_name=bytes(patch_name,'utf8'))
                        
                        writer.write(tf_example_obj.SerializeToString())
                        seq_number += 1

    # prototype a return report - what is really needed?
    svs_file_conversion_dict = {'mask_dict': mask_dict, 
                                'tfrecord_file_name': tfrecord_file_name, 
                                'number_of_patches': seq_number, 
                                'temp_dir': temp_dir}
    
    return svs_file_conversion_dict


def get_tfrecord_marked_thumbnail(tfrecord_filename, wsi_filename, thumb_scale, border_color='red'):
    """ create and mark a thumbnail image with the TFRecord file patch loacations
    
    Args:
        tfrecord_filename:  TensorFlow TFRecord file as created by svs_file_to_patches_tfrecord()
        wsi_filename:       Whole Scale Image filename compatible with OpenSlide
        thumb_scale:        WSI reduction scale to define thumbnail image size
        border_color:       string: red, green, blue, brown, yellow, white, black, orange, tan 
    Returns:
        marked_thumbnail:   PIL Image with tfrecord image locations marked
        
    """
    # open the WSI - get FSI size & get thumbnail, make a grayscale copy as "RGBA" & calculate scale
    os_obj = openslide.OpenSlide(wsi_filename)
    pixels_height = np.int(os_obj.dimensions[1] * thumb_scale)
    pixels_width = np.int(os_obj.dimensions[0] * thumb_scale)
    one_thumb = os_obj.get_thumbnail((pixels_height, pixels_width)).convert('RGBA')    
    one_thumb_draw = ImageDraw.Draw(one_thumb)
    
    iterable_tfrecord = get_iterable_tfrecord(tfrecord_filename).__iter__()
    is_empty = False
    while is_empty == False:
        try:
            patch_record = iterable_tfrecord.next()
            image_raw = patch_record['image_raw'].numpy()
            ulc_row = np.int(patch_record['ulc_row'].numpy() * thumb_scale)
            ulc_col = np.int(patch_record['ulc_col'].numpy() * thumb_scale)
            lrc_row = np.int(patch_record['lrc_row'].numpy() * thumb_scale)
            lrc_col = np.int(patch_record['lrc_col'].numpy() * thumb_scale)
            one_thumb_draw.rectangle(((ulc_col, ulc_row), (lrc_col, lrc_row)), outline=border_color, fill=None)
            
        except StopIteration:
            is_empty = True
            pass
    
    return one_thumb


def get_tfrecord_masked_thumbnail(tfrecord_filename, wsi_filename, thumb_scale, alpha, border_color='blue'):
    """ create a grayscale thumbnail image and insert scaled TFRecord file patch images
    
    Args:
        tfrecord_filename:  TensorFlow TFRecord file as created by svs_file_to_patches_tfrecord()
        wsi_filename:       Whole Scale Image filename compatible with OpenSlide
        thumb_scale:        WSI reduction scale to define thumbnail image size
        alhpa:              (0, 1) - blend the grayscale with the patch
        border_color:       string: red, green, blue, brown, yellow, white, black, orange, tan
        
    Returns:
        masked_thumbnail:   PIL Image with tfrecord image locations marked
    """
    # open the WSI - get FSI size & get thumbnail, make a grayscale copy as "RGBA" & calculate scale
    os_obj = openslide.OpenSlide(wsi_filename)
    pixels_height = np.int(os_obj.dimensions[1] * thumb_scale)
    pixels_width = np.int(os_obj.dimensions[0] * thumb_scale)
    
    # get the height and width of the first TFRecord patch
    iterable_tfrecord = get_iterable_tfrecord(tfrecord_filename).__iter__()
    patch_record = iterable_tfrecord.next() 
    patch_height = np.int(patch_record['lrc_row'].numpy()) - np.int(patch_record['ulc_row'].numpy()) + 1
    patch_width = np.int(patch_record['lrc_col'].numpy()) - np.int(patch_record['ulc_col'].numpy()) + 1
    
    # get the mask, thumnail image and boxes scaling arrays
    thumbnail_divisor = 1 / thumb_scale
    mask_dict = get_mask_w_scale_grid(os_obj, patch_height, patch_width, thumbnail_divisor)
    
    # extract the thumbnail image and mask
    one_thumb = mask_dict['one_thumb'].convert('RGBA')
    one_gray_thumb = one_thumb.convert('L')
    one_gray_thumb = one_gray_thumb.convert('RGBA')
    black_mask = mask_dict['thumb_mask'].convert('RGBA')
    
    # also see PIL.Image:  composite
    thumb_mask = PIL.Image.blend(one_gray_thumb, black_mask, alpha)
    if border_color is None:
        one_thumb = PIL.Image.blend(one_thumb, one_gray_thumb, alpha)
    else:
        one_thumb = PIL.Image.blend(one_thumb, black_mask, alpha)
    
    # get the full list of patches
    iterable_tfrecord = get_iterable_tfrecord(tfrecord_filename).__iter__()
    is_empty = False
    while is_empty == False:
        try:
            # get & scale the tfrecord patch
            patch_record = iterable_tfrecord.next()
            image_raw = patch_record['image_raw'].numpy()
            ulc_row = np.int(patch_record['ulc_row'].numpy() * thumb_scale)
            ulc_col = np.int(patch_record['ulc_col'].numpy() * thumb_scale)
            lrc_row = np.int(patch_record['lrc_row'].numpy() * thumb_scale)
            lrc_col = np.int(patch_record['lrc_col'].numpy() * thumb_scale)
            box_bound = (ulc_col, ulc_row, lrc_col, lrc_row)
            
            # get the patch image & add a border
            patch_im = one_thumb.crop(box_bound).convert('RGBA')
            if not border_color is None:
                patch_draw = ImageDraw.Draw(patch_im)
                h = patch_im.size[0]
                w = patch_im.size[1]
                patch_draw.rectangle(((0, 0), (h-1, w-1)), outline=border_color, fill=None)
            
            # paste the scaled & bordered tfrecord image patch into the thumbnail
            offset = (ulc_col, ulc_row)
            thumb_mask.paste(patch_im, offset)

        except StopIteration:
            is_empty = True
            pass
    
    return thumb_mask
