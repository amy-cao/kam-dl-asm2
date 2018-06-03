from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2

from config import *


def build_resnet_model():
    main_input = Input(shape=MODEL_INPUT_SHAPE)

    x = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), kernel_regularizer=l2(REG_CONST))(main_input)

    x = conv_block(x, filters=[16, 16, 32], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[16, 16, 32], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[16, 16, 32], kernel_size=(3, 3), regularizer=l2(REG_CONST))

    x = conv_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))

    x = conv_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[32, 32, 64], kernel_size=(3, 3), regularizer=l2(REG_CONST))

    x = conv_block(x, filters=[64, 64, 128], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[64, 64, 128], kernel_size=(3, 3), regularizer=l2(REG_CONST))
    x = residual_block(x, filters=[64, 64, 128], kernel_size=(3, 3), regularizer=l2(REG_CONST))

    x = Flatten()(x)
    main_output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(REG_CONST), use_bias=True)(x)

    return Model(inputs=main_input, outputs=main_output)


def conv_block(block_input, filters, kernel_size, regularizer, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    x = Conv2D(filters=filters1
             , kernel_size=(1, 1)
             , strides=strides
             , kernel_regularizer=regularizer)(block_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters2
             , kernel_size=kernel_size
             , padding='same'
             , kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters3
             , kernel_size=(1, 1)
             , kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters=filters3
                    , kernel_size=(1, 1)
                    , strides=strides
                    , kernel_regularizer=regularizer)(block_input)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def residual_block(block_input, filters, kernel_size, regularizer):
    filters1, filters2, filters3 = filters
    x = Conv2D(filters=filters1
             , kernel_size=(1, 1)
             , kernel_regularizer=regularizer)(block_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters2
             , kernel_size=kernel_size
             , padding='same'
             , kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters3
             , kernel_size=(1, 1)
             , kernel_regularizer=regularizer)(x)

    x = layers.add([x, block_input])
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    return x


if __name__ == '__main__':
    model = build_resnet_model()
    model.summary()
