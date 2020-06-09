import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils.generic_utils import CustomObjectScope

from Networks.resfcn256_6 import resfcn256_6

from Networks import mobilenet_v2


class PosPrediction_6():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256_6(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_op, 6])
        self.x_op = self.network(self.x, is_training=False)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    def restore(self, model_path):
        tf.train.Saver(self.network.vars).restore(self.sess, model_path)

    def predict(self, image):
        pos = self.sess.run(self.x_op,
                            feed_dict={self.x: image[np.newaxis, :, :, :]})
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op,
                            feed_dict={self.x: images})
        return pos * self.MaxPos


class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is some what of amystery..
        self.model = None

        # set tensorflow session GPU usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    def restore(self, model_path):
        with CustomObjectScope(
                {'relu6': mobilenet_v2.relu6}):  # ,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
            self.model = keras.models.load_model(model_path)

    def predict(self, image):
        x = image[np.newaxis, :, :, :]
        pos = self.model.predict(x=x)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError


class PosPrediction_6_keras():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is some what of amystery..
        self.model = None

        # set tensorflow session GPU usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    def restore(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, image):
        pos = self.model.predict(x=image[np.newaxis, :, :, :])
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError
