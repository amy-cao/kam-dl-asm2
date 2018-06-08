import numpy as np
import os

import image_utils
from config import *

class DataAugmentor:
    ''' Argument for __init__: True or False valued flag indicating to perform the transformation or not'''
    def __init__(self,
                left_shift=False, right_shift=False, up_shift=False, down_shift=False,
                anticw_rotate=False, cw_rotate=False,
                left_shear=False, right_shear=False,
                scale=False):
        self.left_shift = left_shift
        self.right_shift = right_shift
        self.up_shift = up_shift
        self.down_shift = down_shift

        self.anticw_rotate = anticw_rotate
        self.cw_rotate = cw_rotate

        self.left_shear = left_shear
        self.right_shear = right_shear

        self.scale = scale


    def save_augmented_images(self, images, labels):
        if self.anticw_rotate:
            augmented = self.apply_transformation(images, image_utils.random_anticw_rotate)
            self.save_images_and_labels(augmented, labels, 'anticw_rotate')
        if self.cw_rotate:
            augmented = self.apply_transformation(images, image_utils.random_cw_rotate)
            self.save_images_and_labels(augmented, labels, 'cw_rotate')

        if self.left_shift:
            augmented = self.apply_transformation(images, image_utils.random_left_shift)
            self.save_images_and_labels(augmented, labels, 'left_shift')
        if self.right_shift:
            augmented = self.apply_transformation(images, image_utils.random_right_shift)
            self.save_images_and_labels(augmented, labels, 'right_shift')
        if self.up_shift:
            augmented = self.apply_transformation(images, image_utils.random_up_shift)
            self.save_images_and_labels(augmented, labels, 'up_shift')
        if self.down_shift:
            augmented = self.apply_transformation(images, image_utils.random_down_shift)
            self.save_images_and_labels(augmented, labels, 'down_shift')

        if self.left_shear:
            augmented = self.apply_transformation(images, image_utils.random_left_shear)
            self.save_images_and_labels(augmented, labels, 'left_shear')
        if self.right_shear:
            augmented = self.apply_transformation(images, image_utils.random_right_shear)
            self.save_images_and_labels(augmented, labels, 'right_shear')

        if self.scale:
            augmented = self.apply_transformation(images, image_utils.shrink_img)
            self.save_images_and_labels(augmented, labels, 'scale')


    def save_images_and_labels(self, images, labels, folder_name):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        counter = 0
        filename = '{}.txt'.format(folder_name)
        filename = os.path.join(DATA_DIR, filename)
        f = open(filename, 'w')
        for image, label in zip(images, labels):
            image_name = 'img{0:03d}-{1:05d}.png'.format(label+1, counter)
            img_path = os.path.join(folder_path, image_name)
            image_utils.save_img(image, img_path)
            counter += 1
            # save txt file for labels
            assert isinstance(label, np.int_)
            line = '{} {}\n'.format(image_name, label)
            f.write(line)
        f.close()


    def apply_transformation(self, images, fn):
        transformed_images = np.copy(images)
        for i, image in enumerate(images):
            transformed_images[i] = fn(image)
        return transformed_images


if __name__ == '__main__':
    images, labels = image_utils.load_train_data()
    data_aug = DataAugmentor(left_shift=False, right_shift=False, up_shift=False, down_shift=False,
                            anticw_rotate=False, cw_rotate=False,
                            left_shear=False, right_shear=False,
                            scale=False)
    data_aug.save_augmented_images(images, labels)
