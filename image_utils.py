import os
import random
from collections import Counter


import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import utils
from config import *


def load_labels_dict(path):
    ''' Reads the given path-label lookup file into a dictionary '''
    lookup_dict = dict()
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            k, v = line.split(' ')
            lookup_dict[k] = int(v.rstrip())

    return lookup_dict


def load_data(data_path, label_path):
    ''' Given data path and labels path, read images and its labels into ndarray '''
    X, y = [], []
    labels_map = load_labels_dict(label_path)
    for img_path, label in labels_map.items():
        img = load_img(os.path.join(data_path, img_path))
        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)


def load_train_data():
    return load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH)


def load_val_data():
    return load_data(VAL_DATA_PATH, VAL_LABEL_PATH)


def load_train_and_val_data():
    ''' Returns (X_train, y_train), (X_val, y_val) as np arrays '''
    return load_train_data(), load_val_data()


def load_img(path):
    ''' Given img path, read img into ndarray with 1 channel '''
    with Image.open(path) as pil_img:
        return np.array(pil_img)[:, :, None]    # Get the colour dimension


def display_img(img):
    if img is None or len(img.shape) != 3:
        print('display_img: the input image must be not None and must have 3 dimensions')
    else:
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.show()


def rotated_img(img, degree):
    ''' Return a rotated img in numpy '''
    assert img is not None
    # Add white pixels at the boundaries
    rotated = ndi.interpolation.rotate(img, degree, mode='constant', cval=255)
    pil_img = Image.fromarray(rotated.squeeze().astype('uint8')).resize(img.shape[:2])    # Only need w/h dimension
    rotated = np.array(pil_img)[:, :, None]     # Expand the channel dimension
    return rotated


def random_rotation(img, lower=-45, upper=45):
    degree = np.random.randint(lower, upper + 1)
    return rotated_img(img, degree)


def check_class_num(labels):
    return dict(Counter(labels))



if __name__ == '__main__':
    images, labels = load_data(VAL_DIRECTORY)
    print(labels.shape)

    # images = np.expand_dims(images, axis=-1)
    # img = images[1]
    # new_img = rotate_and_shear(img,40,45,0,0)
    # # new_img = rotate_and_shear(img,0,0,40,45)
    # pyplot.imshow(img[:,:,0], cmap='gray')
    # pyplot.show()
    # pyplot.imshow(new_img[:,:,0], cmap='gray')
    # pyplot.show()
