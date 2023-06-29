import random

from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.transforms import Flip
from albumentations.augmentations.blur.transforms import GaussianBlur
from albumentations.augmentations.transforms import GaussNoise


class NotSoLongCrop(RandomCropFromBorders):
    def __init__(self, ratio, min_side, always_apply=False, p=1.0):
        super().__init__(0,0,0,0,always_apply,p)
        self.ratio = ratio
        self.min_side = min_side

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        while True:
            x_min = random.randint(0, img.shape[1] - self.min_side - 1)
            x_max = random.randint(x_min + self.min_side, img.shape[1] - 1)
            y_min = random.randint(0, img.shape[0] - self.min_side - 1)
            y_max = random.randint(y_min + self.min_side, img.shape[0] - 1)

            if self.ratio < (x_max - x_min) / (y_max - y_min) < 1 / self.ratio:
                break

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
