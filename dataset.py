import os
from PIL import Image
from matplotlib import pyplot
import numpy as np
import random
from collections import Counter
import scipy.ndimage as ndi
from keras.preprocessing.image import ImageDataGenerator
from config import *


# vali: each class 101 examples
# train: each class 611 examples


def read_images(directory):
    image_names = os.listdir(directory) # list of image name
    images = []
    labels = []
    for img in image_names:
        name = os.path.join(directory, img)
        im_frame = Image.open(name)
        np_frame = np.array(im_frame.getdata()).reshape(128,128)
        images.append(np_frame)

        label = int(img[4:6])-1
        labels.append(label)
    return np.array(images), np.array(labels)


# TODO
def rotate_and_shear(image, rot_lo, rot_up, sh_lo, sh_up):
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


    transform_matrix = None
    # rotate
    deg = np.random.uniform(rot_lo, rot_up)
    theta = np.deg2rad(deg)
    if deg:
        rotate_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
        transform_matrix = rotate_matrix

    # shear
    # strength = np.random.uniform(sh_lo, sh_up)
    # shear = np.deg2rad(sh_up)
    # if shear:
    #     shear_matrix = np.array([[1, -np.sin(shear), 0],
    #                              [0, np.cos(shear), 0],
    #                              [0, 0, 1]])
    #     transform_matrix = shear_matrix #if transform_matrix is None else np.dot(transform_matrix, shear_matrix)


    # apply transform
    row_axis, col_axis, channel_axis, fill_mode, cval = 1,2,0,'nearest',0

    if transform_matrix is not None:
        h, w = image.shape[row_axis], image.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        image = np.rollaxis(image, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in image]
        image = np.stack(channel_images, axis=0)
        image = np.rollaxis(image, 0, channel_axis + 1)
    return image


def check_class_num(labels):
    return dict(Counter(labels))


def shuffle_data(images, labels):
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    return images, label


if __name__ == '__main__':
    images, labels = read_images(VAL_DIRECTORY)
    print(labels.shape)
    
    # images = np.expand_dims(images, axis=-1)
    # img = images[1]
    # new_img = rotate_and_shear(img,40,45,0,0)
    # # new_img = rotate_and_shear(img,0,0,40,45)
    # pyplot.imshow(img[:,:,0], cmap='gray')
    # pyplot.show()
    # pyplot.imshow(new_img[:,:,0], cmap='gray')
    # pyplot.show()
