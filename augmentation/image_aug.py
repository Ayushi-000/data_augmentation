# augmentation/image_aug.py

import cv2
import numpy as np
import albumentations as A

class ImageAugmentor:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=30, p=0.5),
        ])

    def augment(self, image_path, save_path):
        image = cv2.imread(image_path)
        augmented = self.transform(image=image)['image']
        cv2.imwrite(save_path, augmented)
        return save_path
