"""
@authors:
* Austin Dibble, University of Glasgow

Utility for calculating the voxel intensity histograms of a dataset.

Example (in container): python ./brain/utilities/vol_histograms.py 
    --csv_path /brain_age/output/2024-08-02_17-52-08_model=sxogyn5q_test_T1_ADNI/final.csv 
    --col X_test --outdir /brain_age/output/ --dataset_name ADNI

"""


import argparse
import os
from pathlib import Path
from time import localtime, strftime
import random


import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some CSV, column and output directory.')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to the input CSV file.')
    parser.add_argument('--col', required=True, type=str, help='Column name in the CSV file containing paths to .nii.gz files.')
    parser.add_argument('--outdir', required=True, type=str, help='Directory where the results will be stored.')
    parser.add_argument('--dataset_name', default='ADNI', type=str, help='Name of the dataset')
    parser.add_argument('--num_vols', default=1000, type=int, help='Directory where the results will be stored.')
    parser.add_argument('--images_path', required=False, type=str, help='Provide the root path of the ADNI image data to use a different data folder.')

    return parser.parse_args()

def create_output_folder(base_output_dir, dataset_name):
    local_time_str = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    out_dir = Path(base_output_dir, f"{local_time_str}_voxel_dists_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def load_volume_paths(csv_path, column_name, images_path=None):
    df = pd.read_csv(csv_path)

    df[column_name].replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    def update_path(file_path):
        base_filename = Path(file_path).name  # Get the base filename
        new_full_path = images_path / base_filename  # Prepend the new path
        new_full_path_str = str(new_full_path)
        new_full_path_str = new_full_path_str.replace('___', '_')
        new_full_path_str = new_full_path_str.replace('__', '_')
        return new_full_path_str

    if images_path:
        df[column_name] = df[column_name].apply(update_path)

    return df[column_name].tolist()


def z_scoring(x):
    ''' Function to z-score the data taking into account only non-zero voxels. '''

    # Find mean and std of non-zero voxels
    non_zero_x = ~np.isclose(x, 0)
    mn = np.mean(x[non_zero_x])
    std = np.std(x[non_zero_x])

    # Z-score only non-zero voxels
    y = np.copy(x)
    y[~non_zero_x] = 0
    y[non_zero_x] = (y[non_zero_x] - mn) / std
    return y

def calculate_histogram(volume_path, bin_edges, z_bin_edges):
    img = nib.load(volume_path)
    data = img.get_fdata()
    data = data[data > 0] # filter out zero voxels
    hist, _ = np.histogram(data, bins=bin_edges)

    z_data = z_scoring(data)
    z_hist, _ = np.histogram(z_data, bins=z_bin_edges)

    return hist, z_hist

def calculate_bin_edges(vol_paths):
        # Determine the global min and max values across all volumes
    global_min = float('inf')
    global_max = float('-inf')

    global_z_min = float('inf')
    global_z_max = float('-inf')
    
    for volume_path in vol_paths:
        img = nib.load(volume_path)
        data = img.get_fdata()
        data = data[data > 0]  # Filter out non-positive values
        if data.size > 0:
            global_min = min(global_min, data.min())
            global_max = max(global_max, data.max())

            z_data = z_scoring(data)
            global_z_min = min(global_z_min, z_data.min())
            global_z_max = max(global_z_max, z_data.max())

    # Create bin edges from global min to global max
    bin_edges = np.linspace(global_min, global_max, 1001)
    z_bin_edges = np.linspace(global_z_min, global_z_max, 1001)

    return bin_edges, z_bin_edges

def sample_paths(paths_list, sample_size=1000):
    """
    Samples up to sample_size random paths from the paths list.
    
    Parameters:
    paths_list (list): List of paths to sample from.
    sample_size (int): The maximum number of paths to sample. Default is 1000.
    
    Returns:
    list: A list of sampled paths.
    """
    if len(paths_list) <= sample_size:
        return paths_list
    else:
        return random.sample(paths_list, sample_size)

def main():
    args = parse_arguments()
    
    csv_path = args.csv_path
    column_name = args.col
    output_dir = args.outdir
        
    out_dir = create_output_folder(output_dir, args.dataset_name)
    
    logger.info('Loading CSV paths')
    volume_paths = load_volume_paths(csv_path, column_name, Path(args.images_path))
    
    logger.info('Sampling paths from CSV')
    volume_paths = sample_paths(volume_paths, sample_size=args.num_vols)
    
    logger.info('Calculate bin edges from global min/max')
    bin_edges, z_bin_edges = calculate_bin_edges(volume_paths)

    logger.info('Adding up voxel histograms')
    histograms = []
    zscore_histograms = []
    for volume_path in volume_paths:
        hist, z_hist = calculate_histogram(volume_path, bin_edges, z_bin_edges)
        histograms.append(hist)

        zscore_histograms.append(z_hist)
    
    mean_histogram = np.mean(histograms, axis=0)
    mean_z_histogram = np.mean(zscore_histograms, axis=0)
    
    def save_hist(hist, bin_edges, file:str):

        # Plot the mean histogram
        plt.figure(figsize=(10, 6))
        plt.plot(bin_edges[:-1], hist)  # Use bin edges for x-axis        
        plt.title('Mean Voxel Intensity Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save the plot
        plot_path = out_dir / file
        plt.savefig(plot_path)
        
        print(f'Mean histogram plot saved to {plot_path}')

        np.save(out_dir / f'{plot_path.stem}.npy', hist)
        np.save(out_dir / f'{plot_path.stem}_edges.npy', np.asarray(bin_edges))

    logger.info('Plotting')
    save_hist(mean_histogram, bin_edges, file='mean_hist.png')
    save_hist(mean_z_histogram, z_bin_edges, file='mean_z_hist.png')

    logger.info('Done')

if __name__ == '__main__':
    main()
