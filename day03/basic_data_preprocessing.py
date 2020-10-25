from basic_transforms import *


class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        # add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose([
            RandomScale(),
            RandomFlip(),
            Pad(image_size, mean_val=[0.485, 0.456, 0.406]),
            RandomCrop(image_size)
            , ConvertDataType(),
            Normalize(mean_val, std_val)
        ])

    def __call__(self, image, label):
        return self.augment(image, label)
