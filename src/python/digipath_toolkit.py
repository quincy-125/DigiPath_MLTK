import os
import sys

import numpy as np

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
    if patch_image_name_dict['file_type'][0] != '.':
        patch_image_name_dict['file_type'] = '.' + patch_image_name_dict['file_type']
        
    patch_name = patch_image_name_dict['case_id']
    patch_name += '_%i'%patch_image_name_dict['location_x']
    patch_name += '_%i'%patch_image_name_dict['location_y'] 
    patch_name += '_%s'%patch_image_name_dict['class_label']
    patch_name += '%s'%patch_image_name_dict['file_type']
    
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
                                 'file_type': '.' + field[4] }
    """
    name_type_list = patch_file_name.strip().split('.')
    name_field_list = name_type_list[0].split('_')
    
    patch_image_name_dict = {'case_id': name_field_list[0], 
                             'location_x': int(name_field_list[1]), 
                             'location_y': int(name_field_list[2]), 
                             'class_label': name_field_list[3], 
                             'file_type': '.' + name_type_list[-1]}
    
    return patch_image_name_dict