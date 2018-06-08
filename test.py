import sys
from keras.models import load_model
from keras.utils import to_categorical

import utils
from config import *
from image_utils import load_val_data


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 test.py <model to test>')
        exit()

    model = load_model(sys.argv[1])
    X_test, y_test = utils.preprocess(*load_val_data())
    print('Test Loss', 'Test Accuracy')
    print(model.evaluate(X_test, y_test))
