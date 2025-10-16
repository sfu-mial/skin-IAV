import random

import torchvision.transforms.functional as trF_func
from torchvision.transforms import InterpolationMode


class RandomRotate90(object):
    """
    Transform to rotate an image to a desired angle of rotation as specified.
    Inspired by: https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/transforms.py
    """

    def __init__(self, expand=False, center=None):
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0, 3)
        angle = idx * 90
        return angle

    def __call__(self, image):
        angle = self.get_params()
        img = trF_func.rotate(
            img=image,
            angle=angle,
            expand=self.expand,
            center=self.center,
            interpolation=InterpolationMode.BILINEAR,
        )
        return img
