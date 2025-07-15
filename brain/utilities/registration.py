"""

Created on Tuesday - September 03 2024

@authors:
* Austin Dibble, University of Glasgow

Utility functions for nifti volume registration to template volumes.

"""

from pathlib import Path

import ants


def load_volume(vol_path: Path, orient: str = None):
    """
    Loads an ANTS image object from the volume path. Re-orients the volume to the given space.

    Args:
        vol_path: Path object or str where the volume is on disk. .nii.gz file.
        orient: A string like 'LIA', or 'RAS' for the new orientation space. If None, then it does not re-orient.

    Returns: ANTS volume array object.
    """
    return ants.image_read(vol_path, reorient=orient)


def load_template_volume(template_path: str = None, orient: str = None):
    """
    Loads an ANTS image template object. If template_path is None, then the default MNI template is returned.

    Args:
        template_path: str or Path from where to load the template. If None, the default is loaded.
        orient: A string like 'LIA', or 'RAS' for the new orientation space. If None, then it does not re-orient.

    Returns: MNI volume array object, if found.
    """
    if not template_path:
        template_path = ants.get_ants_data("mni")
    mni = ants.image_read(template_path, reorient=orient)
    return mni


def get_registration_transform(img, template):
    """
    Generates a registration transform for an ANTS image to an ANTS template.
    Both must be given. The transform is returned. Can be applied with `apply_registration_transform()`.

    Args:
        img: ANTS image array object.
        template: ANTS image array object for template. Can be retrieved using `load_template_volume()`.

    Returns: ANTS transform array object.
    """
    # MNI registration: T1w
    transformation = ants.registration(
        fixed=template,  # template
        moving=img,  # image to register
        type_of_transform="SyN",
        verbose=False,
    )

    return transformation


def apply_registration_transform(target, transform):
    """
    Given a target volume and transform object, this applies the transform to the given target volume.
    Returns the registered target volume.

    Args:
        target: ANTS volume array object to be transformed.
        transform: ANTS transform array object. Can be obtained from `get_registration_transform()`.

    Returns: Registered array volume object.
    """
    registered_target = ants.apply_transforms(
        moving=target,  # register the 'i_error_map'
        fixed=transform["warpedmovout"],
        transformlist=transform["fwdtransforms"],
        verbose=False,
    )
    return registered_target


def to_numpy(img):
    """
    Converts an ANTS volume object to a numpy array.

    Args:
        img: ANTS volume object.

    Returns: numpy array of volume
    """
    return img.numpy()


def from_numpy(img_arr, ref_vol=None):
    """
    Creates an ANTS volume array object from a numpy volume. 
    If a ref_vol is given, then that information is used to conform the given numpy array.

    Args:
        img_arr: 3D numpy array representing the brain volume.
        ref_vol: A reference volume to conform the numpy array.

    Returns:
        Loaded ANTS volume array object.
    """
    if ref_vol:
        return ants.from_numpy(
            img_arr, spacing=ref_vol.spacing, origin=ref_vol.origin, direction=ref_vol.direction
        )  # Convert to ANTs image
    else:
        return ants.from_numpy(img_arr)


def to_file(img, path_out):
    """
    Saves the given ANTS volume array to disk at path_out. Does not create the full path if it doesn't exist.

    Args:
        img: ANTS volume array object
        path_out: Path or str for where to save the volume.

    Returns:
        none
    """
    img.to_file(path_out)
