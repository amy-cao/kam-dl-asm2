from config import *

from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

def build_resnet_model():
    main_input = Input(shape=MODEL_INPUT_SHAPE)
    x = conv_block(main_input, 7, 3, l2(6e-5))

    x = residual_block(x, 32, 3, l2(6e-5))
    x = residual_block(x, 32, 3, l2(6e-5))

    x = residual_block(x, 64, 3, l2(6e-5))
    x = residual_block(x, 64, 3, l2(6e-5))

    x = residual_block(x, 96, 3, l2(6e-5))
    x = residual_block(x, 96, 3, l2(6e-5))

    x = residual_block(x, 128, 3, l2(6e-5))
    x = residual_block(x, 128, 3, l2(6e-5))
    
    x = residual_block(x, 256, 3, l2(6e-5))
    x = residual_block(x, 256, 3, l2(6e-5))

    main_output = Dense(NUM_CLASSES)
    model = Model(inputs=[main_input], outputs=[policy, value])

    return model


def residual_block(block_input, filters, kernel_size, regularizer, padding='valid', use_bias=False):
    x = Conv2D(filters
             , kernel_size
             , padding=padding
             , use_bias=use_bias
             , kernel_regularizer=regularizer)(block_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters
             , kernel_size
             , padding=padding
             , use_bias=use_bias
             , kernel_regularizer=regularizer)(block_input)
    x = BatchNormalization()(x)

    x = add([block_input, x])
    x = Activation('relu')(x)

    return x































# can load pre trained model in keras as well
def vgg16(x):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    return x
