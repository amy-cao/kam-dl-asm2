import os

DIRECTORY = '../data/'
TRAIN_LABEL = 'train.txt'
VAL_LABEL = 'vali.txt'

TRAIN_DIRECTORY = os.path.join(DIRECTORY, 'train-set/')
VAL_DIRECTORY = os.path.join(DIRECTORY, 'vali-set/')
TMP_DIRECTORY = os.path.join(DIRECTORY, 'tmp-set/')

NUM_CLASSES = 62
MODEL_INPUT_SHAPE = (128, 128)

NUM_RESIDUAL_BLOCKS = 6
