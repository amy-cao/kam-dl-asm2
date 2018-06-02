from image_utils import load_train_data
from config import *

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam

# images, labels = read_images(TMP_DIRECTORY)


images, labels = load_train_data()
labels = to_categorical(labels, num_classes=NUM_CLASSES)
print(images.shape)
print(labels.shape)
# images shape: (# examples, 128, 128)
# labels shape: (# examples, number of classes)

main_input = Input(shape=MODEL_INPUT_SHAPE)
x = Conv2D(filters=32, kernel_size=3, padding='valid', strides=2, activation='relu')(main_input)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=3, padding='valid', strides=2, activation='relu')(x)
x = BatchNormalization()(x)

x = Flatten()(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(main_input, output)
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(images, labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)
model.save('../saved_models/base_model_bz32_ep100_acc_valacc.h5')
