import time
import sys
import os

import random
import numpy as np
import torch

from model.implicit_networks import UNet3D

from einops import rearrange, reduce, repeat
from monai.transforms import (
    ScaleIntensityRange,
    AddChannel,
    SqueezeDim,
    ToTensor,
    ToNumpy,
    AsDiscrete,
    KeepLargestConnectedComponent,
    CastToType
)
from monai.transforms import Compose
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

from natsort import natsorted

import SimpleITK as sitk


def save_itk_new(image, filename, origin, spacing, direction):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image_new(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyDirection


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InnerTransform(object):
    def __init__(self):
        self.ToNumpy = ToNumpy()
        self.AsDiscrete05 = AsDiscrete(threshold=0.5)
        self.AsDiscrete08 = AsDiscrete(threshold=0.8)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.AddChannel = AddChannel()
        self.SqueezeDim = SqueezeDim()
        self.ToTensorFloat32 = ToTensor(dtype=torch.float)
        self.ToTensorUINT = ToTensor(dtype=torch.uint8)


InnerTransformer = InnerTransform()