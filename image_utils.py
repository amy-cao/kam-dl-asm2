import os
import math
import random
from collections import Counter

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage import transform
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


def save_img(img, path):
    ''' Save given img to given path '''
    verify_img(img)
    assert image.shape == MODEL_INPUT_SHAPE, 'Image to be saved must have the default dimension (128, 128, 1)'
    pil_img = Image.fromarray(img.squeeze().astype('uint8'))
    pil_img.save(path)


def display_img(img):
    if img is None or len(img.shape) != 3:
        print('display_img: the input image must be not None and must have 3 dimensions')
    else:
        plt.imshow(img[:, :, 0], cmap='gray', vmin=0, vmax=255)
        plt.show()


def verify_img(img):
    assert img is not None, 'Image cannot be None'
    assert len(img.shape) == 3, 'Image must have 3 dimensions, got {}'.format(img.shape)
    assert img.max() <= 255, 'All pixels should have value <= 255'


def shift_img(img, x_shift, y_shift):
    verify_img(img)
    shifted = ndi.interpolation.shift(img, (y_shift, x_shift, 0), mode='constant', cval=255)
    assert img.shape == shifted.shape
    return shifted


def rotate_img(img, degree):
    ''' Return a rotated img in numpy '''
    verify_img(img)
    # Add white pixels at the boundaries
    rotated = ndi.interpolation.rotate(img, degree, mode='constant', cval=255)
    pil_img = Image.fromarray(rotated.squeeze().astype('uint8')).resize(img.shape[:2])    # Only need w/h dimension
    rotated = np.array(pil_img)[:, :, None]     # Expand the channel dimension
    return rotated


def shear_img(img, degree):
    ''' Return a sheared img in numpy '''
    verify_img(img)
    img2D = img.squeeze()
    tf = transform.AffineTransform(shear=np.deg2rad(degree))
    sheared = transform.warp(img2D, inverse_map=tf, mode='constant', cval=1)
    sheared = (sheared[:, :, None] * 255).astype('uint8')   # Expand dims and cast
    return sheared


def shrink_img(img, factor=DEF_SHRINK):
    ''' Return a scaled img in numpy '''
    verify_img(img)
    assert 0 < factor <= 1 , 'Only support down-scaling'
    new_width, new_height, _ = tuple(int(factor * dim) for dim in img.shape)
    smaller = Image.fromarray(img.squeeze().astype('uint8')).resize((new_width, new_height))
    new_img = Image.new(mode='L', size=img.shape[:2], color=255)
    # Paste the shrinked img in the middle
    top_left = ((img.shape[0] - new_width) // 2,  (img.shape[1] - new_height) // 2)
    new_img.paste(smaller, top_left)
    shrinked = np.array(new_img)[:, :, None]    # Expand the channel dimensions
    return shrinked


def random_shift(img, x_low=0, x_high=0, y_low=0, y_high=0):
    ''' Randomly shift img given bounds '''
    x_shift = np.random.randint(x_low, x_high + 1)
    y_shift = np.random.randint(y_low, y_high + 1)
    return shift_img(img, x_shift, y_shift)

def random_normal_shift(img, x_mean=DEF_X_SHIFT, y_mean=DEF_Y_SHIFT, std=1):
    ''' Randomly shift img given expected shift '''
    x_shift = np.round(np.random.normal(DEF_X_SHIFT, std))
    y_shift = np.round(np.random.normal(DEF_Y_SHIFT, std))
    return shift_img(img, x_shift, y_shift)

def random_left_shift(img, low=MIN_X_SHIFT, high=MAX_X_SHIFT):
    ''' Randomly shift img to left given bound '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shift(img, -high, -low, 0, 0)

def random_right_shift(img, low=MIN_X_SHIFT, high=MAX_X_SHIFT):
    ''' Randomly shift img to right given bound '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shift(img, low, high, 0, 0)

def random_up_shift(img, low=MIN_Y_SHIFT, high=MAX_Y_SHIFT):
    ''' Randomly shift img to top given bound '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shift(img, 0, 0, -high, -low)

def random_down_shift(img, low=MIN_Y_SHIFT, high=MAX_Y_SHIFT):
    ''' Randomly shift img to bottom given bound '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shift(img, 0, 0, low, high)


def random_shear(img, low=0, high=0):
    ''' Randomly shear img given bounds '''
    degree = np.random.randint(low, high + 1)
    return shear_img(img, degree)

def random_normal_shear(img, mean=DEF_SHEAR, std=1):
    degree = np.round(np.random.normal(mean, std))
    return shear_img(img, degree)

def random_left_shear(img, low=MIN_SHEAR, high=MAX_SHEAR):
    ''' Randomly left shear img given bounds '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shear(img, -high, -low)

def random_right_shear(img, low=MIN_SHEAR, high=MAX_SHEAR):
    ''' Randomly right shear img given bounds '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_shear(img, low, high)


def random_rotate(img, low=0, high=0):
    ''' Randomly rotate img given bounds '''
    degree = np.random.randint(low, high + 1)
    return rotate_img(img, degree)

def random_normal_rotate(img, mean=DEF_ROT, std=1):
    ''' Randomly rotate img given expected rotation '''
    degree = np.round(np.random.normal(mean, std))
    return rotate_img(img, degree)

def random_cw_rotate(img, low=MIN_ROT, high=MAX_ROT):
    ''' Randomly rotate img clockwise given bounds '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_rotate(img, -high, -low)

def random_anticw_rotate(img, low=MIN_ROT, high=MAX_ROT):
    ''' Randomly rotate img anticlockwise given bounds '''
    assert 0 <= low <= high, 'low > high or low < 0'
    return random_rotate(img, low, high)


def check_class_num(labels):
    return dict(Counter(labels))


if __name__ == '__main__':
    images, labels = load_val_data()

    # img = load_img('test.png')
    img = images[np.random.choice(len(images))]
    display_img(img)

    temp = random_left_shift(img)
    display_img(temp)

    temp = random_right_shift(img)
    display_img(temp)

    temp = random_up_shift(img)
    display_img(temp)

    temp = random_down_shift(img)
    display_img(temp)

    temp = random_left_shear(img)
    display_img(temp)

    temp = random_right_shear(img)
    display_img(temp)

    temp = random_cw_rotate(img)
    display_img(temp)

    temp = random_anticw_rotate(img)
    display_img(temp)
    # shrinkimg = shrink_img(img, 0.9)
    # display_img(shrinkimg)
    # print(shrinkimg.dtype)
    # #
    # shiftimg = random_shift(img)
    # shiftimg = random_upshift(img)
    # display_img(shiftimg)
    # print(shiftimg.dtype)
    # #
    # rotateimg = rotate_img(img, 45)
    # display_img(rotateimg)
    # print(rotateimg.dtype)
    #
    # shearimg = shear_img(img, 11)
    # display_img(shearimg)
    # print(shearimg.dtype)
    #
    #
    # shearrotateimg = shear_img(rotate_img(img, -10), -20)
    # display_img(shearrotateimg)
    #
    # shiftrotateimg = shift_img(rotate_img(img, -10), 20, 0)
    # display_img(shiftrotateimg)
    #
    # shrinkrotateimg = shrink_img(rotate_img(img, -10), 0.8)
    # display_img(shrinkrotateimg)

    # print(img.shape)
    # # print(shiftimg.shape)
    # # print(rotateimg.shape)
    # print(shearimg.shape)
    #
    # img = np.ones((128, 128, 1))
    # img[:] = 128
    # img = img.astype('uint8')
    # # img2D = img.squeeze()
    # # tf = transform.AffineTransform(shear=np.deg2rad(10))
    # # sheared = transform.warp(img2D, inverse_map=tf)[:, :, None]
    # sheared = shear_img(img, -20)
    # display_img(img)
    # display_img(sheared)


    # new_img = rotate_and_shear(img,40,45,0,0)
    # # new_img = rotate_and_shear(img,0,0,40,45)
    # pyplot.imshow(img[:,:,0], cmap='gray')
    # pyplot.show()
    # pyplot.imshow(new_img[:,:,0], cmap='gray')
    # pyplot.show()
