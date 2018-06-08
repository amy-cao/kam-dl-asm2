import numpy as np
import datetime
from keras.utils import to_categorical

from config import *


def cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stress_message(message, extra_newline=False):
    print('{2}{0}\n{1}\n{0}{2}'.format('='*len(message), message, '\n' if extra_newline else ''))


def shuffle_data(a, b):
    ''' Shuffles 2 np arrays with same length together '''
    assert len(a) == len(b)                 # Sanity check
    random_state = np.random.get_state()    # Store random state s.t. 2 shuffles are the same
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)
    np.random.seed()    # Re-seed generator


def standardise(data):
    if len(data.shape) != 4 or data.shape[1:] != DEF_IMG_SHAPE:
        raise ValueError('Wrong dimension! (n, 128, 128, 1) required')
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    data = (data - mean) / np.maximum(stddev, 1e-15)
    return data


def preprocess(images, labels):
    # TODO: Probably shift to a better preprocessing
    # Naive preprocessing: zero mean and unit range
    images = images.astype('float64')
    images /= 127.5
    images -= 1
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    return images, labels
