
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import ants

def load_volume(vol_path:Path, orient:str=None):
    return ants.image_read(vol_path, reorient=orient)

def load_template_volume(template_path:str=None, orient:str=None):
    if not template_path:
        template_path = ants.get_ants_data('mni')
    mni = ants.image_read(template_path, reorient=orient)
    return mni

def get_registration_transform(img, template):
    # MNI registration: T1w
    transformation = ants.registration(fixed = template,                 # template
                                        moving = img,               # image to register
                                        type_of_transform = 'SyN',
                                        verbose = False)   
    
    return transformation

def apply_registration_transform(target, transform):
    registered_target = ants.apply_transforms(moving = target,           # register the 'i_error_map'
                                        fixed = transform['warpedmovout'],
                                        transformlist = transform['fwdtransforms'],
                                        verbose = False)
    return registered_target

def to_numpy(img):
    return img.numpy()

def from_numpy(img_arr, ref_vol=None):
    if ref_vol:
        return ants.from_numpy(img_arr,
                                spacing = ref_vol.spacing,
                                origin = ref_vol.origin, 
                                direction = ref_vol.direction)               # Convert to ANTs image
    else:
        return ants.from_numpy(img_arr)

def to_file(img, path_out):
    img.to_file(path_out)