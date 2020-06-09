"""
    Most of code by Xiaochus https://github.com/xiaochus/MobileNetV2
"""

"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

from contextlib import redirect_stdout

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Add, DepthwiseConv2D
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.models import Model


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def decoder_network(se):
    pd = Conv2DTranspose(512, 4, strides=1, padding='same', activation=K.relu)(se)  # 8 x 8 x 512
    pd = Conv2DTranspose(256, 4, strides=2, padding='same', activation=K.relu)(pd)  # 16 x 16 x 256
    pd = Conv2DTranspose(256, 4, strides=1, padding='same', activation=K.relu)(pd)  # 16 x 16 x 256
    pd = Conv2DTranspose(256, 4, strides=1, padding='same', activation=K.relu)(pd)  # 16 x 16 x 256
    pd = Conv2DTranspose(128, 4, strides=2, padding='same', activation=K.relu)(pd)  # 32 x 32 x 128
    pd = Conv2DTranspose(128, 4, strides=1, padding='same', activation=K.relu)(pd)  # 32 x 32 x 128
    pd = Conv2DTranspose(128, 4, strides=1, padding='same', activation=K.relu)(pd)  # 32 x 32 x 128
    pd = Conv2DTranspose(64, 4, strides=2, padding='same', activation=K.relu)(pd)  # 64 x 64 x 64
    pd = Conv2DTranspose(64, 4, strides=1, padding='same', activation=K.relu)(pd)  # 64 x 64 x 64
    pd = Conv2DTranspose(64, 4, strides=1, padding='same', activation=K.relu)(pd)  # 64 x 64 x 64

    pd = Conv2DTranspose(32, 4, strides=2, padding='same', activation=K.relu)(pd)  # 128 x 128 x 32
    pd = Conv2DTranspose(32, 4, strides=1, padding='same', activation=K.relu)(pd)  # 128 x 128 x 32
    pd = Conv2DTranspose(16, 4, strides=2, padding='same', activation=K.relu)(pd)  # 256 x 256 x 16
    pd = Conv2DTranspose(16, 4, strides=1, padding='same', activation=K.relu)(pd)  # 256 x 256 x 16

    pd = Conv2DTranspose(3, 4, strides=1, padding='same', activation=K.relu)(pd)  # 256 x 256 x 3
    pd = Conv2DTranspose(3, 4, strides=1, padding='same', activation=K.relu)(pd)  # 256 x 256 x 3
    pos = Conv2DTranspose(3, 4, strides=1, padding='same', activation=K.sigmoid)(pd)

    return pos


def MobileNetv2_PRN(input_shape, alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)
    if alpha > 1.0:
        last_filters = _make_divisible(512 * alpha, 8)
    else:
        last_filters = 512
    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    output = decoder_network(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model


if __name__ == '__main__':
    model = MobileNetv2_PRN((256, 256, 6), 1.00)
    with open('summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
