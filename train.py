from dataset import read_images, shuffle_data
from config import *

from keras.models import Model
from keras.layers import Input, Conv2D, Dense

images, labels = read_images(TMP_DIRECTORY)

# images shape: (# examples, 128, 128)
# labels shape: (# examples,)

main_input = Input(shape=MODEL_INPUT_SHAPE)
