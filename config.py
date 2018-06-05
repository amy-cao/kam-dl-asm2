import os

''' Misc '''
DEF_IMG_SIZE = (128, 128)
DEF_IMG_SHAPE = (*DEF_IMG_SIZE, 1)


''' Data Augmentation '''
DEF_SHRINK = 0.85

DEF_ROT = 10            # Degrees
MIN_ROT = 7.5
MAX_ROT = 12.5

DEF_SHEAR = 11.25       # Degrees
MIN_SHEAR = 9
MAX_SHEAR = 14

DEF_X_SHIFT = 10        # Pixels
MIN_X_SHIFT = 8
MAX_X_SHIFT = 12

DEF_Y_SHIFT = 5         # Pixels
MIN_Y_SHIFT = 4
MAX_Y_SHIFT = 6


''' Path '''
DATA_DIR = '../data/'
TRAIN_LABEL = 'train.txt'
VAL_LABEL = 'vali.txt'

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train-set/')
TRAIN_LABEL_PATH = os.path.join(DATA_DIR, TRAIN_LABEL)
VAL_DATA_PATH = os.path.join(DATA_DIR, 'vali-set/')
VAL_LABEL_PATH = os.path.join(DATA_DIR, VAL_LABEL)

TMP_DIR = os.path.join(DATA_DIR, 'tmp-set/')


''' Model '''
MODEL_INPUT_SHAPE = DEF_IMG_SHAPE
NUM_CLASSES = 62
REG_CONST = 2e-4

''' Training '''
NUM_VAL_DATA = 3100
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.01
LR_SEARCH = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

