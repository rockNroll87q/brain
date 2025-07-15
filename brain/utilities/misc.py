"""
Created on 2024-07-02

@authors:
* Austin Dibble, University of Glasgow

Miscellaneous utility functions in brain library
"""

import numpy as np


def pad_volume_to_shape(volume, target_shape=(256, 256, 256)):
    """
    Given a volume, adds necessary constant padding to yield desired shape.
    Does NOT handle if the volume is *larger* than the desired shape. Only smaller!
    """
    # padding = [(0, max(target_dim - current_dim, 0)) for target_dim, current_dim in zip(target_shape, volume.shape)]

    padding = []
    for current_dim, target_dim in zip(volume.shape, target_shape, strict=True):
        total_padding = target_dim - current_dim
        # Split padding evenly between 'before' and 'after', with extra padding added to 'after' if odd
        before_padding = total_padding // 2
        after_padding = total_padding // 2 + total_padding % 2  # Add the extra padding to 'after' if odd
        padding.append((before_padding, after_padding))

    padded_vol = np.pad(volume, padding, mode="constant", constant_values=0)
    return padded_vol, padding


def parse_predicted_variable(value):
    """Given a incoming argument string like "[var1,var2,var3]", this function splits them into a list."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        # Attempt to parse a string representation of a list
        try:
            # Removing brackets and splitting by comma
            return value.strip("[]").split(",")
        except ValueError:
            # If parsing fails, return the string as is
            return value
    else:
        return value
