import os
import numpy as np
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization

from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import utils
import image_utils
from config import *
from image_utils import load_train_data, load_val_data, load_data
from model import build_resnet_model


def train_data_generator(images, labels, batch_size):
    num_batches = len(images) // batch_size
    while True:
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            if i == num_batches - 1:
                end = len(images)

            image_batch, label_batch = images[start:end], labels[start:end]
            image_batch, label_batch = utils.preprocess(image_batch, label_batch)
            yield image_batch, label_batch


def load_all_images_and_labels(use_augmentation=True):
    # List of all images/labels, in the form of np arrays
    all_images, all_labels = [], []

    images, labels = load_train_data()
    all_images.append(images)
    all_labels.append(labels)

    if use_augmentation:
        # Load augmentations; assumes the augmentations have been generated
        augmentations = ['left_shift', 'right_shift', 'up_shift', 'down_shift', 'cw_rotate', 'anticw_rotate', 'left_shear', 'right_shear', 'scale']

        for aug in augmentations:
            images, labels = load_data(os.path.join(DATA_DIR, aug), os.path.join(DATA_DIR, '{}.txt'.format(aug)))
            all_images.append(images)
            all_labels.append(labels)

    # Stack all np arrays into 1
    images = np.vstack(all_images)
    labels = np.hstack(all_labels)

    return images, labels


def train():
    # Get Data
    images, labels = load_all_images_and_labels()

    # Shuffle
    utils.shuffle_data(images, labels)

    print()
    print('Total number of examples:', len(images))

    # Split for validation
    train_images, val_images = np.split(images, [-NUM_VAL_DATA])
    train_labels, val_labels = np.split(labels, [-NUM_VAL_DATA])

    val_images, val_labels = np.copy(val_images), np.copy(val_labels)
    val_images, val_labels = utils.preprocess(val_images, val_labels)

    print('Number of training examples:', len(train_images))
    print('Number of validation examples:', len(val_images))
    print()


    # Grid search
    learning_rates = sorted(set([LEARNING_RATE, *LR_SEARCH]))
    batch_sizes = [BATCH_SIZE, BATCH_SIZE//2, BATCH_SIZE//4, BATCH_SIZE*2, BATCH_SIZE*4]

    for lr in learning_rates:
        for bs in batch_sizes:
            message = 'At {}, Training model with LR {} and Batch Size {}'.format(utils.cur_time(), lr, bs)
            utils.stress_message(message, extra_newline=True)

            # Create data generator: preprcessing is done in the generator
            datagen = train_data_generator(train_images, train_labels, bs)

            # Build model and train
            model = build_resnet_model()

            model.compile(optimizer=SGD(lr=lr, momentum=0.9, nesterov=True)
                        , loss='categorical_crossentropy'
                        , metrics=['accuracy'])

            model.fit_generator(datagen
                              , steps_per_epoch=len(train_images)//bs
                              , epochs=NUM_EPOCHS
                              , validation_data=(val_images, val_labels)
                              , callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1)
                                         , EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=8, verbose=1)
                                         , ModelCheckpoint('checkpoints/resnet-lr{}-bs{}-epoch{{epoch:02d}}-val_loss{{val_loss:.3f}}-val_acc{{val_acc:.2f}}.h5'.format(lr, bs)
                                                          , save_best_only=True, verbose=1)]
                              , shuffle=True)

            model.save('../saved_models/resnet_v4_lr{}_bs{}.h5'.format(lr, bs))

            # Load test data
            X_test, y_test = utils.preprocess(*load_val_data())
            test_loss, test_acc = model.evaluate(X_test, y_test)
            utils.stress_message('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc), True)


if __name__ == '__main__':
    train()


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
