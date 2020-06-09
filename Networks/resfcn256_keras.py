from keras import backend as keras_backend
from keras import layers
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from keras.models import Model


# from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64
def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = LeakyReLU()(y)

    return y


class resfcn256_keras(object):
    def __init__(self, resolution_inp=256, resolution_op=256, base_filters=16, in_channels=6, out_channels=3):
        if (resolution_inp != 256) or (resolution_op != 256):
            print("resolution input or output is not 256, model generation might fail..")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.base_filters = 16
        self.model = self.get_keras_model()

    def get_keras_model(self):
        input_shape = (self.resolution_inp, self.resolution_inp, self.in_channels)
        inputs = Input(shape=input_shape)

        # encoder
        x_e = Conv2D(self.base_filters, 4, strides=1, padding='same')(inputs)  # 256 x 256 x 16

        x_e = residual_block(x_e, self.base_filters * 2, _strides=(2, 2))  # 128 x 128 x 32
        x_e = residual_block(x_e, self.base_filters * 2, _strides=(1, 1))  # 128 x 128 x 32

        x_e = residual_block(x_e, self.base_filters * 4, _strides=(2, 2))  # 64 x 64 x 64
        x_e = residual_block(x_e, self.base_filters * 4, _strides=(1, 1))  # 64 x 64 x 64

        x_e = residual_block(x_e, self.base_filters * 8, _strides=(2, 2))  # 32 x 32 x 128
        x_e = residual_block(x_e, self.base_filters * 8, _strides=(1, 1))  # 32 x 32 x 128

        x_e = residual_block(x_e, self.base_filters * 16, _strides=(2, 2))  # 16 x 16 x 256
        x_e = residual_block(x_e, self.base_filters * 16, _strides=(1, 1))  # 16 x 16 x 256

        x_e = residual_block(x_e, self.base_filters * 32, _strides=(2, 2))  # 8 x 8 x 512
        x_e = residual_block(x_e, self.base_filters * 32, _strides=(1, 1))  # 8 x 8 x 512

        # decoder
        x_d = Conv2DTranspose(self.base_filters * 32, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_e)  # 8 x 8 x 512

        x_d = Conv2DTranspose(self.base_filters * 16, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(
            x_d)  # 16 x 16 x 256
        x_d = Conv2DTranspose(self.base_filters * 16, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 16 x 16 x 256
        x_d = Conv2DTranspose(self.base_filters * 16, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 16 x 16 x 256

        x_d = Conv2DTranspose(self.base_filters * 8, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(
            x_d)  # 32 x 32 x 128
        x_d = Conv2DTranspose(self.base_filters * 8, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 32 x 32 x 128
        x_d = Conv2DTranspose(self.base_filters * 8, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 32 x 32 x 128

        x_d = Conv2DTranspose(self.base_filters * 4, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(
            x_d)  # 64 x 64 x 64
        x_d = Conv2DTranspose(self.base_filters * 4, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 64 x 64 x 64
        x_d = Conv2DTranspose(self.base_filters * 4, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 64 x 64 x 64

        x_d = Conv2DTranspose(self.base_filters * 2, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(
            x_d)  # 128 x 128 x 32
        x_d = Conv2DTranspose(self.base_filters * 2, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 128 x 128 x 32

        x_d = Conv2DTranspose(self.base_filters * 1, 4, strides=(2, 2), padding='same', activation=keras_backend.relu)(
            x_d)  # 256 x 256 x 16
        x_d = Conv2DTranspose(self.base_filters * 1, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(
            x_d)  # 256 x 256 x 16

        x_d = Conv2DTranspose(self.out_channels, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_d)
        x_d = Conv2DTranspose(self.out_channels, 4, strides=(1, 1), padding='same', activation=keras_backend.relu)(x_d)
        x_d = Conv2DTranspose(self.out_channels, 4, strides=(1, 1), padding='same', activation=keras_backend.sigmoid)(
            x_d)
        model = Model(inputs, x_d)
        return model


if __name__ == "__main__":
    network = resfcn256_keras()
    print(network.model.summary())
