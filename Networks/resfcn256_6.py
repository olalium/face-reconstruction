import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
             scope=None):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                  activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256_6(object):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=6, name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16
                    # x: 256 x 256 x 6
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1)  #
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  #
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  #
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  #
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  #
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  #
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  #
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  #
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  #
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  #
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 640
                    pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd, 3, 4, stride=1,
                                               activation_fn=tf.nn.sigmoid)  # , padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

                    return pos

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
