#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Monday - November 24 2025, 12:31:14

@author: Michele Svanera, University of Glasgow

The code is intended to create segmentation masks for brain images.
It takes in input an image or a directory of images and produces segmentation masks as output.
It is possible to specify the input and output directories, as well as other parameters as the 
desidered segmentation masks (i.e., which tasks and which methods per task).
All the volumes are expected to be in NIfTI format and preprocessed.
A list of segmentation methods is available below: 'Methods_available'.
Tasks:
- defacing
- skull stripping
- Left / right hemisphere segmentation
- aseg
- aseg+aparc

'''


## Imports

import os, sys
import argparse
from os.path import join as opj
import concurrent.futures
from datetime import timedelta
import time


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import subprocess


## Paths and Constants

Path_in = '../in/'
Path_out = '../out/'
Methods_available = {'defacing': ['method1', 'method2'],
                     'skull_stripping': ['method1', 'method2'],
                     'left_right_hemisphere': ['method1', 'method2'],
                     'aseg': ['method1', 'method2'],
                     'aseg_aparc': ['method1', 'method2']}


## Functions

def findListOfAnatomical(path_in, identifier='nii.gz'):
    """A function to find all anatomical files in a given directory and its subdirectories."""

    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + '/' + i_file)
    
    return sorted(list(np.unique(all_anat)))

def run_bash_command(bash_command, i_subj_id):
    """ Run a bash command via subprocess module """

    # print(f'Running {bash_command} for {i_subj_id}')
    process = subprocess.run(bash_command.split(),
                    stdout=subprocess.PIPE,
                    text=True)
    if process.returncode != 0:         # errors
        print(f'Error with: {i_subj_id}')
        return 1
    return 0

def thread_function(i_vol):
    x = 'echo Hello World'
    bash_command = f"{x} -o {x} -f {x} -z y {x}"
    if run_bash_command(bash_command, x):
        return
    return


## Main

def main(all_vols):


    # Check how many processes the node can handle





    # Get all the volumes to process
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = executor.map(thread_function, 
                        all_vols)


    # Write a short summary of the results
    
    

    return


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Create segmentation masks for brain images.')
    parser.add_argument('--input', type=str, default=Path_in, help='Path to the input directory of brain images or single volume.')  # noqa: E501
    parser.add_argument('--output_dir', type=str, default=Path_out, help='Path to the output directory to save segmentation masks.')  # noqa: E501
    parser.add_argument('--log_fullpath', type=str, default='segmentation_masks_creation.log', help='Path to the log file.')  # noqa: E501
    parser.add_argument('--tasks', type=str, nargs='+', default=['defacing', 'skull_stripping', 'left_right_hemisphere', 'aseg', 'aseg_aparc'], help='List of segmentation tasks to perform.')  # noqa: E501
    parser.add_argument('--methods_per_task', type=str, nargs='+', help='List of methods to use per task.')
    
    args = parser.parse_args()
    path_in = args.input
    path_out = args.output_dir
    tasks = args.tasks
    methods_per_task = args.methods_per_task

    # Checks tasks and method availabilities
    assert len(tasks) > 0, 'No tasks specified. Please provide at least one task.'
    assert all([task in Methods_available for task in tasks]), 'Some tasks are not recognized. Available tasks are: ' + str(list(Methods_available.keys()))  # noqa: E501
    assert len(methods_per_task) > 0, 'No methods specified. Please provide at least one method per task.'
    for i_task in tasks:
            for i_method in methods_per_task:
                assert i_method in Methods_available[i_task], f'Method {i_method} is not recognized for task {i_task}. Available methods are: ' + str(Methods_available[i_task])  # noqa: E501
    
    # Get the list of all the volumes to process
    all_vols = findListOfAnatomical(path_in)
    assert len(all_vols) > 0, 'No anatomical volumes found in the input path.'
    print(f'Found {len(all_vols)} volumes to process.')

    # Start processing
    start_time = time.time()

    main(all_vols)

    time_needed = timedelta(seconds=(time.time() - start_time))
    print('Done! [Tot: ' + str(time_needed) + ']')