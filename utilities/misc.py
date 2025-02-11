"""
Created on 2024-07-02

@authors:
* Austin Dibble, University of Glasgow

Miscellaneous utility functions in brain library
"""

import numpy as np

def pad_volume_to_shape(volume, target_shape=(256,256,256)):
    # padding = [(0, max(target_dim - current_dim, 0)) for target_dim, current_dim in zip(target_shape, volume.shape)]

    padding = []
    for current_dim, target_dim in zip(volume.shape, target_shape):
        total_padding = target_dim - current_dim
        # Split padding evenly between 'before' and 'after', with extra padding added to 'after' if odd
        before_padding = total_padding // 2
        after_padding = total_padding // 2 + total_padding % 2  # Add the extra padding to 'after' if odd
        padding.append((before_padding, after_padding))

    padded_vol = np.pad(volume, padding, mode='constant', constant_values=0)
    return padded_vol, padding