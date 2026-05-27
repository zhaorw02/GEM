import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
np.bool = np.bool_
import imgaug.augmenters as iaa
from PIL import Image


# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        # Execute one of the following noise augmentations
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.02*255), per_channel=0.5
            ),
            iaa.AdditiveLaplaceNoise(scale=(0.0, 0.02*255), per_channel=0.5),
            iaa.AdditivePoissonNoise(lam=(0.0, 0.02*255), per_channel=0.5)
        ]),
        iaa.Sometimes(
            0.168,
            iaa.OneOf([
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.5)),
                    iaa.AverageBlur(k=(3, 5)),
                    iaa.MedianBlur(k=(3, 5)),
                ]),
                iaa.MotionBlur(k=(5, 9)),
            ]),
        ),
        iaa.Sometimes(
            0.168,
            # Higher values denote stronger compression
            iaa.JpegCompression(compression=(0, 50))
        ),
    ],
    # do all of the above augmentations in random order
    random_order=True
)


def image_corrupt(image: Image):
    image_arr = np.array(image)
    image_arr = image_arr[None, ...]
    
    image_arr = seq(images=image_arr)
    
    image = Image.fromarray(image_arr[0])
    return image
