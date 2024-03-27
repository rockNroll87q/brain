#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to augment training data.
"""

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
import albumentations as albu
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from brain.utilities.config import AugmentConfig
from scipy import stats
from loguru import logger
from typing import Tuple, Union, Optional


def addWeighted(src1, alpha, src2, beta, gamma):
    """ 
    Calculates the weighted sum of two arrays (cv2 replaced).

    :param src1: first input array.
    :param aplha: weight of the first array elements.
    :param src2: second input array of the same size and channel number as src1.
    :param beta: weight of the second array elements.
    :param gamma: scalar added to each sum
    :return: output array that has the same size and number of channels as the input arrays.
    """

    return src1 * alpha + src2 * beta + gamma


def augmentation_salt_and_pepper_noise(X_data, generator:np.random.Generator, amount=10. / 1000):
    """ 
    Function to add S&P noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param amount: quantity of voxels affected
    :return X_data_out: augmented volume
    """

    X_data_out = X_data
    salt_vs_pepper = 0.2  # Ration between salt and pepper voxels
    n_salt_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * salt_vs_pepper))
    n_pepper_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * (1.0 - salt_vs_pepper)))

    # Add Salt noise
    coords = [generator.integers(0, i - 1, int(n_salt_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.max(X_data)

    # Add Pepper noise
    coords = [generator.integers(0, i - 1, int(n_pepper_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.min(X_data)

    return X_data_out


class SaltAndPepperNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p=1.0):
        super(SaltAndPepperNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        return augmentation_salt_and_pepper_noise(img, self.rng)


def augmentation_gaussian_noise(X_data, generator:np.random.Generator):
    """ 
    Function to add gaussian noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :return X_data_out: augmented volume
    """

    # Gaussian distribution parameters
    X_data_no_background = X_data
    mean = np.mean(X_data_no_background)
    var = np.var(X_data_no_background)
    sigma = var ** 0.5

    gaussian = generator.normal(mean, sigma, X_data.shape).astype(X_data.dtype)

    # Compose the output (src1, alpha, src2, beta, gamma)
    X_data_out = addWeighted(X_data, 0.8, gaussian, 0.2, 0)

    return X_data_out


class GaussianNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p=1.0):
        super(GaussianNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        return augmentation_gaussian_noise(img, self.rng)


def augmentation_inhomogeneity_noise(X_data, inhom_vol, generator):
    """ 
    Function to add inhomogeneity noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param inhom_vol: inhomogeneity volume (preloaded)
    :return X_data_out: augmented volume
    """

    # Randomly select a vol of the same shape of 'X_data'
    x_1 = generator.integers(0, int(X_data.shape[0]) - 1, size=1)[0]
    x_2 = generator.integers(0, int(X_data.shape[1]) - 1, size=1)[0]
    x_3 = generator.integers(0, int(X_data.shape[2]) - 1, size=1)[0]
    y_1 = inhom_vol[x_1: x_1 + X_data.shape[0],
          x_2: x_2 + X_data.shape[1],
          x_3: x_3 + X_data.shape[2]]

    # Compose the output: add noise to the original vol
    X_data_out = X_data + y_1.astype(X_data.dtype)

    return X_data_out


class InhomogeneityNoiseAugment(ImageOnlyTransform):

    def __init__(self, inhom_vol: np.array, seed=None, always_apply=False, p=1.0):
        super(InhomogeneityNoiseAugment, self).__init__(always_apply, p)
        self.inhom_vol = inhom_vol
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        return augmentation_inhomogeneity_noise(img, self.inhom_vol, self.rng)



def fast_change_luminance_contrast(X_data, generator:np.random.Generator, clipping=False, threshold=0.025, by_slice=True):
    gamma = (3.0 - 0.5) * generator.random() + 0.5

    # Calculate min and max values differently based on the value of by_slice
    if by_slice:
        # Min and Max calculated along specific dimensions
        X_min, X_max = X_data.min(axis=(0, 1), keepdims=True), X_data.max(axis=(0, 1), keepdims=True)
    else:
        # Min and Max calculated over the entire volume
        X_min, X_max = X_data.min(), X_data.max()
        # Ensure X_min and X_max have the same shape as expected in further operations
        X_min, X_max = X_min.reshape((1, 1, 1)), X_max.reshape((1, 1, 1))

    X_norm = (X_data - X_min) / (X_max - X_min + 0.001)

    if clipping:
        X_norm[X_data < threshold] = 0

    X_gamma = X_norm ** (1.0 / gamma)
    X_data_out = X_gamma * (X_max - X_min) + X_min

    return X_data_out

class GammaNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p_by_slice=0.5, p=1.0):
        super(GammaNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)
        self.p_by_slice = p_by_slice
    
    def apply(self, img, **params):
        if self.rng.random() < self.p_by_slice:
            by_slice = True
        else:
            by_slice = False

        return fast_change_luminance_contrast(img, self.rng, by_slice=by_slice)

def change_contrast(X_data: np.ndarray, generator:np.random.Generator, min_alpha: float=0.5, max_alpha: float=3.0):
    """ 
    Function to change the contrast of the input volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param min_alpha: min value for the contrast
    :param max_alpha: max value for the contrast
    :return X_data_out: augmented volume
    """

    X_min, X_max = X_data.min(), X_data.max()
    alpha = (max_alpha - min_alpha) * generator.random() + min_alpha       # Contrast
    beta = 0                                                                                    # Brightness

    # Apply contrast change to 'X_data'
    X_data_out = np.clip((alpha * X_data + beta), X_min, X_max)

    return X_data_out


class ContrastNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p=1.0):
        super(ContrastNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)
        
    def apply(self, img, **params):
        return change_contrast(img, self.rng)


def slice_spacing(X_data: np.ndarray, generator:np.random.Generator, min_slice_rep: int=2, max_slice_rep: int=5):
    """ 
    Function to add more slices of the input volume in Axial view.
    If 'slice_repetitions'=2, means every slice is repeated twice (for a total of 2, consecutive).
    

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param min_slice_rep: min amount of consecutive slices in Axial view
    :param max_slice_rep: max amount of consecutive slices in Axial view
    :return X_data_out: augmented volume
    """

    slice_repetitions = generator.integers(min_slice_rep, max_slice_rep)

    # Apply contrast change to 'X_data'
    X_data_out = X_data[:, ::(slice_repetitions), :]                    # keep only '256/(slice_repetitions)' slices
    X_data_out = np.repeat(X_data_out, slice_repetitions, axis=1)       # repeat the slice 'slice_repetitions' times
    X_data_out = X_data_out[:, :X_data.shape[2], :]                     # take the same shape as the beginning

    assert X_data_out.shape == X_data.shape

    return X_data_out


class SliceSpacingNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p=1.0):
        super(SliceSpacingNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)
    
    def apply(self, img, **params):
        return slice_spacing(img, self.rng)


def augmentation_bias_noise(X_data: np.ndarray, generator:np.random.Generator):
    """ 
    Function to add bias noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :return X_data_out: augmented volume
    """
    
    # Extract a value for 'n_cycles' from 1 to 7, and a possible transpose choice
    n_cycles = generator.integers(1, 7)
    possible_transpose = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    i_choice = generator.choice(range(len(possible_transpose)))
    Factor = 2.

    # Create sin wave
    x = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    y = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    z = np.linspace(-np.pi * n_cycles/Factor, np.pi * n_cycles/Factor, 256)
    xx, yy, zz = np.meshgrid(x, y, z)
    noise = np.sin(np.transpose(xx, axes=(0,2,1)) + yy + zz) + \
            np.sin(xx + np.transpose(yy, axes=(0,2,1)) + zz) + \
            np.sin(xx + yy + np.transpose(zz, axes=(0,2,1)))

    # Change (transpose) the axes to add variability
    noise = np.transpose(noise, axes=possible_transpose[i_choice])          

    # Adjust the amplitude of the noise (1/10 of the volume)
    noise = stats.zscore(noise, axis=None)
    noise = noise / 2

    # Add noise to avoid easy filter detection
    mean = np.mean(noise)
    var = np.var(noise)
    sigma = var ** 0.5
    gaussian = generator.normal(mean, sigma, noise.shape).astype(noise.dtype)
    noise = addWeighted(noise, 0.9, gaussian, 0.1, 0)        # Compose the output (src1, alpha, src2, beta, gamma)

    # Apply it only on non-zero, or near to zero, of 'X_data' (i.e., min after z-scoring)
    noise[X_data == X_data.min()] = 0
    
    # Compose the output: add noise to the original vol
    X_data_out = X_data + noise.astype(X_data.dtype)

    return X_data_out


class BiasNoiseAugment(ImageOnlyTransform):
    def __init__(self, seed=None, p=1.0):
        super(BiasNoiseAugment, self).__init__(always_apply=False, p=p)
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        return augmentation_bias_noise(img, self.rng)

class FastBiasNoiseAugment(ImageOnlyTransform):
    def __init__(self, max_cycles=5, factor=2.0, shape=(256, 256, 256), seed=None, p=1.0):
        super(FastBiasNoiseAugment, self).__init__(always_apply=False, p=p)
        """
        Pre-compute noise volumes for bias noise augmentation.

        :param max_cycles: Maximum number of cycles for sine wave generation.
        :param factor: Scaling factor for the wave.
        :param shape: Shape of the volume for which noise is precomputed.
        :param seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.noise_volumes = self._precompute_noise_volumes(max_cycles, factor, shape)
        self.translate = FastTranslationAugment(max_shift=[80, 80, 80], padding_mode='wrap', p=0.5)

    def _precompute_noise_volumes(self, max_cycles, factor, shape):
        """
        Precompute all possible noise volumes based on sine waves.

        :param max_cycles: Maximum number of cycles for sine wave generation.
        :param factor: Scaling factor for the wave.
        :param shape: Shape of the volume for which noise is precomputed.
        :return: Precomputed noise volumes.
        """
        logger.debug(f'Precomputing {max_cycles} bias volumes for FastBiasNoiseAugment')
        noise_volumes = []
        for n_cycles in range(1, max_cycles + 1):
            x = np.linspace(-np.pi * n_cycles/factor, np.pi * n_cycles/factor, shape[0])
            y = np.linspace(-np.pi * n_cycles/factor, np.pi * n_cycles/factor, shape[1])
            z = np.linspace(-np.pi * n_cycles/factor, np.pi * n_cycles/factor, shape[2])
#             xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#             noise = np.sin(xx + yy + zz) + np.sin(xx - yy - zz)
            xx, yy, zz = np.meshgrid(x, y, z)
            noise = np.sin(np.transpose(xx, axes=(0,2,1)) + yy + zz) + \
                    np.sin(xx + np.transpose(yy, axes=(0,2,1)) + zz) + \
                    np.sin(xx + yy + np.transpose(zz, axes=(0,2,1)))
            noise_volumes.append(noise)
        return np.stack(noise_volumes)

    def apply(self, img, **params):
        """
        Apply a randomly selected pre-computed bias noise volume to the input image.

        :param img: Input volume (3D) with shape matching the precomputed noise.
        :return: Augmented volume.
        """

        # Randomly select one of the precomputed noise patterns
        idx = self.rng.integers(0, len(self.noise_volumes))
        noise = self.noise_volumes[idx]

        noise = self.translate.apply(noise, **self.translate.get_params())
        
        if self.rng.random() < 0.5:
            noise = np.flip(noise, axis=self.rng.choice([0, 1, 2]))
        if self.rng.random() < 0.5:
            noise = np.flip(noise, axis=self.rng.choice([0, 1, 2]))

        # Adjust the amplitude of the noise (1/10 of the volume)
        noise = stats.zscore(noise, axis=None)
        noise = noise / 2

        # Add Gaussian noise to avoid easy filter detection
        mean = np.mean(noise)
        sigma = np.std(noise)
        gaussian = self.rng.normal(mean, sigma, noise.shape)
        noise = addWeighted(noise, 0.9, gaussian, 0.1, 0)

        # It may be that the noise volume is larger than the input image, so we need to crop it back to match the size of the input
        if noise.shape != img.shape:
            noise = noise[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]

        # Apply it only on non-zero, or near to zero, of 'img' (i.e., min after z-scoring)
        mask = img != img.min()
        img_out = img.copy()
        img_out[mask] += noise[mask].astype(img.dtype)

        return img_out

def translate_volume(image,
                     shift_x0: int, shift_x1: int, shift_x2: int,
                     padding_mode: str = 'nearest',
                     spline_interp_order: int = 1):
    """ 
    Function to apply volume translation to a single volume.

    :param image: input volume (3D) -> shape (x,y,z)
    :param shift_x0-shift_x1-shift_x2: shift in voxels
    :param padding_mode: the padding mode
    :param spline_interp_order: order for the affine transformation
    :return: augmented volume
    """

    # Set the affine transformation matrix
    M_t = np.eye(4)
    M_t[:-1, -1] = np.array([-shift_x0, -shift_x1, -shift_x2])

    return affine_transform(image, M_t,
                            order=spline_interp_order,
                            mode=padding_mode,
                            cval=0,
                            output_shape=image.shape)


class TranslationAugment(DualTransform):
    """ Class to deal with translation augmentation. """

    def __init__(self, max_shift: list = [20, 20, 20], always_apply=False, p=1.0):
        super(TranslationAugment, self).__init__(always_apply, p)
        self.max_shift = max_shift

    def get_params(self):

        # Randomly select parameters
        try:
            shifts = [(np.random.RandomState().randint(2 * i) - i) for i in self.max_shift]
            shift_x0, shift_x1, shift_x2 = shifts
        except:
            shift_x0, shift_x1, shift_x2 = [0] * 3

        return {"shift_x0": shift_x0, "shift_x1": shift_x1, "shift_x2": shift_x2}

    def apply(self, img, shift_x0: int = 0, shift_x1: int = 0, shift_x2: int = 0, **params):

        # Apply to image
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode='nearest',
                                       spline_interp_order=1)
        else:
            raise Exception('Error 23: type not supported.')

        return img_out
    
def fast_translate_volume(image, shift_x0: int, shift_x1: int, shift_x2: int, padding_mode: str = 'constant'):
    """Function to apply volume translation to a single volume using simultaneous padding and shifting for efficiency.

    :param image: input volume (3D) with shape (x,y,z)
    :param shift_x0, shift_x1, shift_x2: shift in voxels along each axis
    :param padding_mode: the padding mode ('constant', 'edge', etc.)
    :return: augmented volume
    """

    # Initialize pad widths and slices to keep for each dimension
    pad_widths = [(0, 0)] * 3
    slices_to_keep = [slice(None)] * 3

    # Calculate pad widths and slices to keep based on shift directions
    for i, shift in enumerate([shift_x0, shift_x1, shift_x2]):
        if shift > 0:
            pad_widths[i] = (shift, 0)
            slices_to_keep[i] = slice(0, -shift)
        elif shift < 0:
            pad_widths[i] = (0, -shift)
            slices_to_keep[i] = slice(-shift, None)

    # Apply slicing to get the relevant part of the original volume
    sliced_image = image[tuple(slices_to_keep)]

    # Apply padding based on calculated widths
    if padding_mode == 'constant':
        translated_image = np.pad(sliced_image, pad_widths, mode=padding_mode, constant_values=0)
    else:
        # For other modes like 'edge', 'wrap', etc.
        translated_image = np.pad(sliced_image, pad_widths, mode=padding_mode)

    return translated_image

class FastTranslationAugment(DualTransform): # This is faster than the other translation by about a factor of 10x
    """ Class to deal with translation augmentation. """

    def __init__(self, max_shift: list = [20, 20, 20],  padding_mode='constant', seed=None, always_apply=False, p=1.0):
        super(FastTranslationAugment, self).__init__(always_apply, p)
        self.max_shift = max_shift
        self.padding_mode = padding_mode
        self.rng = np.random.default_rng(seed)

    def get_params(self):

        # Randomly select parameters
        try:
            shifts = [(self.rng.integers(2 * i) - i) for i in self.max_shift]
            shift_x0, shift_x1, shift_x2 = shifts
        except:
            shift_x0, shift_x1, shift_x2 = [0] * 3

        return {"shift_x0": shift_x0, "shift_x1": shift_x1, "shift_x2": shift_x2}

    def apply(self, img, shift_x0: int = 0, shift_x1: int = 0, shift_x2: int = 0, **params):

        # Apply to image
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = fast_translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode=self.padding_mode)
        else:
            raise Exception('Error 23: type not supported.')

        return img_out

class RotationAugment(DualTransform):
    """ Class to deal with rotation augmentation. """

    def __init__(self,
                 max_angle: int = 10,
                 rot_spline_order: int = 3,
                 seed=None,
                 always_apply=False,
                 p=1.0):
        super(RotationAugment, self).__init__(always_apply, p)
        self.max_angle = max_angle
        self.rot_spline_order = rot_spline_order
        self.rng = np.random.default_rng(seed)

    def get_params(self):

        # Randomly select parameters
        random_angle = self.rng.integers(2 * self.max_angle) - self.max_angle
        rot_axes = self.rng.permutation(range(3))[:2]  # random select the 2 rotation axes

        return {"random_angle": random_angle, "rot_axes": rot_axes}

    def apply(self, img, random_angle: int, rot_axes: int, **params):

        # Apply to image
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = rotate(input=img,
                             angle=random_angle,
                             axes=rot_axes,
                             reshape=False,
                             order=self.rot_spline_order,
                             mode='nearest',
                             prefilter=True)
        else:
            raise Exception('Error 24: type not supported.')

        return img_out


class GhostingAugment(ImageOnlyTransform):
    """ Class to deal with ghosting augmentation. """

    def __init__(self,
                 max_repetitions: int = 4,
                 seed=None,
                 always_apply=False,
                 p=1.0):
        super(GhostingAugment, self).__init__(always_apply, p)
        self.max_repetitions = max_repetitions
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        # Randomly select parameters
        repetitions = self.rng.choice(range(1, self.max_repetitions + 1))
        axis = self.rng.choice(range(len(img.shape)))

        img_out = img
        shift_value = 0
        for i_rep in range(1, repetitions + 1):
            # Compute the shift to apply to the data
            shift_value += int(img.shape[axis] / (i_rep + 1))

            # Shift the data and add to the out volume
            data_repetition = np.roll(img, shift_value, axis=axis)
            img_out = addWeighted(img_out, 0.85, data_repetition, 0.15, 0)

        return img_out
    
class RandomMotionAugment(ImageOnlyTransform):
    """ Class to add randomMotion augmentation from https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomMotion """

    def __init__(self, always_apply=False, p=1.0, 
                 degrees = 10, # can also be a tuple of floats
                 translation = 5, # can also be a tuple of floats
                 num_transforms: int = 2, 
                 image_interpolation: str = 'linear', **kwargs):
        super(RandomMotionAugment, self).__init__(always_apply, p)

        try:
            import torchio as tio
        except ImportError as e:
            logger.warning("torchio module is not currently installed. RandomMotionAugment will not have any effect.")
            self.badImport = True
        else:
            self.randMot = tio.RandomMotion(degrees, translation, num_transforms, image_interpolation, **kwargs)
            self.badImport = False

    def apply(self, img, **params):
        if self.badImport:
            return img

        return np.squeeze(self.randMot(np.expand_dims(img, axis=0)), axis=0)
    
class RandomGhostingAugment(ImageOnlyTransform):
    """ Class to add randomMotion augmentation from https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomGhosting """

    def __init__(self, always_apply=False, p=1.0, 
                 num_ghosts:Union[int, Tuple[int, int]] = (1, 6), # can also be a tuple of floats
                 axes = (0, 1, 2), # can also be a tuple of floats
                 intensity:Union[float, Tuple[float, float]] = (0.1, 0.6), 
                 restore:float = 0.02, **kwargs):
        super(RandomGhostingAugment, self).__init__(always_apply, p)

        try:
            import torchio as tio
        except ImportError as e:
            logger.warning("torchio module is not currently installed. RandomGhostingAugment will not have any effect.")
            self.badImport = True
        else:
            self.randGhost = tio.RandomGhosting(num_ghosts, axes, intensity, restore, **kwargs)
            self.badImport = False

    def apply(self, img, **params):
        if self.badImport:
            return img

        return np.squeeze(self.randGhost(np.expand_dims(img, axis=0)), axis=0)

def get_augmentation_by_name(inho_vol, augment: AugmentConfig, name):
    augmentations = {
        "inho": InhomogeneityNoiseAugment(inho_vol, p=augment.prob_inho),
        "rota": RotationAugment(p=augment.prob_rota, max_angle=30, rot_spline_order=1),
        "tran": FastTranslationAugment(padding_mode='wrap', p=augment.prob_tran),
        "blur": albu.Blur(blur_limit=(3, 3), p=augment.prob_blur),
        "salt": SaltAndPepperNoiseAugment(p=augment.prob_salt),
        "gaus": GaussianNoiseAugment(p=augment.prob_gaus),
        "down": albu.OneOf([                                
                    albu.Downscale(scale_min=0.25, scale_max=0.75, interpolation=0, p=1.),
                    albu.Downscale(scale_min=0.25, scale_max=0.75, interpolation=4, p=1.),
                ], p=augment.prob_down),
        "gamm": GammaNoiseAugment(p_by_slice=0.5, p=augment.prob_gamm),
        "cont": ContrastNoiseAugment(p=augment.prob_cont),
        "slic": SliceSpacingNoiseAugment(p=augment.prob_slic),
        "bias": FastBiasNoiseAugment(p=augment.prob_bias),
        "moti": RandomMotionAugment(p=augment.prob_moti),
        "ghos": RandomGhostingAugment(p = augment.prob_ghos)
    }
    return augmentations.get(name)

def get_augm_transforms(inho_vol, augment: AugmentConfig, volume_size: int = 256):
    """
    Get the transformations for volume augmentation.

    :param inho_vol: inhomogeneity volume
    :param volume_size: size of the volume
    :param augment: AugmentConfig class (with all probabilities stored)
    :return: albumentation composition
    """
    if augment.single_type is not None:
        return albu.Compose([
            get_augmentation_by_name(inho_vol, augment, augment.single_type)
        ], p = augment.prob_overall)
    else:
        return albu.Compose([
            # Default transformations
            # albu.VerticalFlip(p = config.augment.prob_flip),  # sagittal plane

            # Geometric transformations
            get_geometric_transforms(augment),
            # Non-geometric transformations
            get_intensity_transforms(inho_vol, augment),

        ], p = augment.prob_overall)
    
def get_geometric_transforms(augment: AugmentConfig):
    """
    Get the geometric/affine transformations for volume augmentation.

    :param augment: AugmentConfig class (with all probabilities stored)
    :return: albumentation composition
    """
    if augment.single_type is not None and augment.single_type in ["rota", "tran"]: # rota and tran are the only two geometric transforms, so we match those
        return albu.Compose([
            get_augmentation_by_name(None, augment, augment.single_type)
        ], p = augment.prob_overall)
    elif not augment.single_type: # If not a single type, return a compose of all geometric transforms
        return albu.Compose([
            # Geometric transformations
            albu.OneOf([
                RotationAugment(p = augment.prob_rota, max_angle=30, rot_spline_order=1),
                FastTranslationAugment(padding_mode='wrap', p = augment.prob_tran),
            ], p = augment.prob_geom),
        ], p=augment.prob_overall)
    else: # Likely this will only happen is it's a single type that isn't in rota or tran
        return None

def get_intensity_transforms(inho_vol, augment:AugmentConfig):
    """
    Get the intensity transformations for volume augmentation.

    :param inho_vol: inhomogeneity volume
    :param augment: AugmentConfig class (with all probabilities stored)
    :return: albumentation composition
    """

    if augment.single_type is not None and augment.single_type not in ["rota", "tran"]: # Opposite to geometric, we want all single transforms which aren't these two
        return albu.Compose([
            get_augmentation_by_name(inho_vol, augment, augment.single_type)
        ], p = augment.prob_overall)
    elif not augment.single_type: # If not a single type, return a compose of all intensity transforms
        return albu.Compose([
            albu.OneOf([
                albu.Blur(blur_limit = (3, 3), p = augment.prob_blur),
                SaltAndPepperNoiseAugment(p = augment.prob_salt),
                GaussianNoiseAugment(p = augment.prob_gaus),
                albu.OneOf([                                # half 'interpolation = 0', half 'interpolation = 4'
                    albu.Downscale(scale_min = 0.25, 
                                    scale_max = 0.75, 
                                    interpolation = 0,      # (cv2.INTER_NEAREST)
                                    p = 1.),
                    albu.Downscale(scale_min = 0.25, 
                                    scale_max = 0.75, 
                                    interpolation = 4,      # (cv2.INTER_LANCZOS4)
                                    p = 1.),
                ], p=augment.prob_down),
                GammaNoiseAugment(p_by_slice=0.5, p = augment.prob_gamm),
                ContrastNoiseAugment(p = augment.prob_cont),
                SliceSpacingNoiseAugment(p = augment.prob_slic),
                FastBiasNoiseAugment(p = augment.prob_bias),   
                InhomogeneityNoiseAugment(inho_vol, p = augment.prob_inho),  # Inhomogeneity noise 
                RandomMotionAugment(p=augment.prob_moti),
                RandomGhostingAugment(p = augment.prob_ghos)
            ], p = augment.prob_colo),
        ], p=augment.prob_overall)
    else: # This should only happen if it's a single type that is rota or tran, so we don't return any augmentations
        return None

class Augmenter: # New augmentation class. Recommended to use this now instead of the above direct functions
    def __init__(self, inho_vol, augment: AugmentConfig):
        self.geom = get_geometric_transforms(augment)
        self.other = get_intensity_transforms(inho_vol, augment)

        # This part is just to ensure that the probabilities are normalized to levels that we expect. 
        # Internally, the normalization is done by albumentations OneOf
        # It doesn't affect the augmentations themselves, but it does ensure that we get a reasonable distribution of types.
        def calculate_weights(desired_probabilities):
            total_probability = sum(desired_probabilities.values())
            normalized_weights = {k: v / total_probability for k, v in desired_probabilities.items()}
            return normalized_weights

        def scale(weights, prob):
            return {k: v * prob for k, v in weights.items()}
        
        augdict = augment.dict()
        non_geo_weights = calculate_weights({k: v for k, v in augdict.items() \
                                    if 'prob' in k and k not in ['prob_overall', \
                                                                 'prob_geom', 'prob_colo', \
                                                                'prob_tran', 'prob_rota']})
        # Scale the non-geometric probabilities by the color augmentation probability and overall probability
        non_geo_weights = scale(non_geo_weights, augment.prob_colo*augment.prob_overall)

        geo_weights = calculate_weights({k: v*augment.prob_geom*augment.prob_overall for k, v in augdict.items() \
                                    if k in ['prob_tran', 'prob_rota']})
        # Scale the geometric probabilities by the overall probability and geometric probability
        geo_weights = scale(geo_weights, augment.prob_overall*augment.prob_geom)

        logger.info(f'Augmentation Normalized Probabilities:')
        logger.info(f'\tGeom Probabilities: {geo_weights}')
        logger.info(f'\tNon-geom Probabilities: {non_geo_weights}')

    def identity(self, image):
        return {'image': image}

    def geometric_augmentation(self, image): # Returns a geometrically transformed volume, if specified. Otherwise, returns image without change
        if self.geom is not None: # Can be none in cases where 'single_type' doesn't include a geometric transform
            return self.geom(image=image)
        
        return self.identity(image)
    
    def intensity_augmentation(self, image): # Returns an intensity-transformed volume, if specified. Otherwise, returns image without change
        if self.other is not None: # Can be none in cases where 'single_type' doesn't include an intensity transform
            return self.other(image=image)
        
        return self.identity(image)

def get_augmenter(inho_vol, augment: AugmentConfig):
    """
    Use this function to get an instance of the Augmenter class for augmenting volumes. Returns None if augmentations aren't active.
    Sample code snippet: 
        dataset = dataset_manager.prepareDataset(config)

        # Create TF datasets for train and valid sets
        augmenter = get_augmenter(dataset['inhomogeneity_volume'], config.augment)

        ds_train, ds_valid, ds_test = dataset_manager.TFDatasetGenerator(config, dataset, augmenter)
        ds_data = {'train': ds_train, 'valid': ds_valid, 'test': ds_test} 
    """
    if augment.augmentation:
        augmenter = Augmenter(inho_vol, augment)
    else:
        augmenter = None

    return augmenter