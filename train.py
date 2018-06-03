import os
import numpy as np
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization

from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import utils
import image_utils
from config import *
from image_utils import load_train_data, load_val_data, load_data
from model import build_resnet_model


# List of all images/labels, in the form of np arrays
all_images, all_labels = [], []

images, labels = load_train_data()
all_images.append(images)
all_labels.append(labels)

# Load augmentations; assumes the augmentations have been generated
augmentations = ['left_shift', 'right_shift', 'up_shift', 'down_shift', 'cw_rotate', 'anticw_rotate', 'left_shear', 'right_shear', 'scale']

for aug in augmentations:
    images, labels = load_data(os.path.join(DATA_DIR, aug), os.path.join(DATA_DIR, '{}.txt'.format(aug)))
    all_images.append(images)
    all_labels.append(labels)

# Stack all np arrays into 1
images = np.vstack(all_images)
labels = np.hstack(all_labels)

# Shuffle
utils.shuffle_data(images, labels)

# Naive zero mean and unit range
images = images.astype('float64')
images /= 127.5
images -= 1

labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split for validation
images, val_images = images[:-NUM_VAL_DATA], images[-NUM_VAL_DATA:]
labels, val_labels = labels[:-NUM_VAL_DATA], labels[-NUM_VAL_DATA:]


model = build_resnet_model()

model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True)
            , loss='categorical_crossentropy'
            , metrics=['accuracy'])

model.summary()

# TODO: Change to fit_generator on the images
model.fit(images, labels
        , batch_size=BATCH_SIZE
        , epochs=NUM_EPOCHS
        , shuffle=True
        , callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=10)])

# Load test data
X_test, y_test = load_val_data()
X_test = X_test.astype('float64')
X_test /= 127.5
X_test -= 1
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

print('Test Loss', 'Test Accuracy')
print(model.evaluate(X_test, y_test, batch_size=BATCH_SIZE))

model.save('../saved_models/resnet_v1.h5')


''' Old model below '''

# datagen = ImageDataGenerator(rotation_range=15
#                            , width_shift_range=0.1
#                            , height_shift_range=0.05
#                            , shear_range=0.2
#                            , zoom_range=0.15)
#
# datagen.fit(images)
#
# # images shape: (# examples, 128, 128, 1)
# # labels shape: (# examples, number of classes)
#
# main_input = Input(shape=MODEL_INPUT_SHAPE)
# x = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(main_input)
# x = BatchNormalization()(x)
# x = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D()(x)
#
# x = Conv2D(filters=48, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Conv2D(filters=48, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D()(x)
#
# x = Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D()(x)
#
# x = Conv2D(filters=80, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = Conv2D(filters=80, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D()(x)
#
# x = Flatten()(x)
# output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(1e-4))(x)
#
# model = Model(main_input, output)
# model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# # model.fit(images, labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
# model.fit_generator(datagen.flow(images, labels, batch_size=BATCH_SIZE)
#                   , steps_per_epoch=len(images) / BATCH_SIZE * 4
#                   , epochs=NUM_EPOCHS
#                   , validation_data=(val_images, val_labels)
#                   , shuffle=True
#                   , use_multiprocessing=True)
#
#
# X_test, y_test = load_val_data()
# X_test = X_test.astype('float64')
# X_test /= 255
# y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
# print(model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE))
#
# model.save('../saved_models/base_model_bz32_ep100_acc_valacc.h5')
