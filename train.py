import numpy as np
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

import utils
from image_utils import load_train_data, load_val_data
from config import *


images, labels = load_train_data()
utils.shuffle_data(images, labels)
print(images.shape)
print(labels.shape)
images = images.astype('float64')
images /= 255
labels = to_categorical(labels, num_classes=NUM_CLASSES)

val_split = 1000
images, val_images = images[:-val_split], images[-val_split:]
labels, val_labels = labels[:-val_split], labels[-val_split:]

datagen = ImageDataGenerator(rotation_range=15
                           , width_shift_range=0.1
                           , height_shift_range=0.05
                           , shear_range=0.2
                           , zoom_range=0.15)

datagen.fit(images)

# images shape: (# examples, 128, 128)
# labels shape: (# examples, number of classes)

main_input = Input(shape=MODEL_INPUT_SHAPE)
x = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(main_input)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=48, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Conv2D(filters=48, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=80, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Conv2D(filters=80, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(1e-4))(x)

model = Model(main_input, output)
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# model.fit(images, labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
model.fit_generator(datagen.flow(images, labels, batch_size=BATCH_SIZE)
                  , steps_per_epoch=len(images) / BATCH_SIZE * 4
                  , epochs=NUM_EPOCHS
                  , validation_data=(val_images, val_labels)
                  , shuffle=True
                  , use_multiprocessing=True)


X_test, y_test = load_val_data()
X_test = X_test.astype('float64')
X_test /= 255
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
print(model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE))

model.save('../saved_models/base_model_bz32_ep100_acc_valacc.h5')
