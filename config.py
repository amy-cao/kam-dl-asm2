import os

DEF_IMG_SIZE = (128, 128)

DATA_DIR = '../data/'
TRAIN_LABEL = 'train.txt'
VAL_LABEL = 'vali.txt'

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train-set/')
TRAIN_LABEL_PATH = os.path.join(DATA_DIR, TRAIN_LABEL)
VAL_DATA_PATH = os.path.join(DATA_DIR, 'vali-set/')
VAL_LABEL_PATH = os.path.join(DATA_DIR, VAL_LABEL)


TMP_DIR = os.path.join(DATA_DIR, 'tmp-set/')


NUM_CLASSES = 62
MODEL_INPUT_SHAPE = (128, 128)

NUM_RESIDUAL_BLOCKS = 6
