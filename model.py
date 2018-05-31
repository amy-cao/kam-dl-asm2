from config import *

def build_resnet_model():
    main_input = Input(shape=MODEL_INPUT_SHAPE)
    x = self.conv_block(main_input, 7, 3, REGULARIZER)

    for _ in range(NUM_RESIDUAL_BLOCKS):
        x = self.residual_block(x, self.filters, 3, REGULARIZER)

    model = KerasModel(inputs=[main_input], outputs=[policy, value])
    return model

def conv_layer(self, layer_input, filters, kernel_size, regularizer):
    return Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    activation="linear",
                    kernel_regularizer=regularizer)(layer_input)


def residual_block(self, block_input, filters, kernel_size, regularizer):
    x = self.conv_layer(block_input, filters, kernel_size, regularizer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = self.conv_layer(x, filters, kernel_size, regularizer)
    x = BatchNormalization()(x)
    x = add([block_input, x])
    x = LeakyReLU()(x)
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
