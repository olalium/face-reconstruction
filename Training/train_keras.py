import argparse
import math
import os
from datetime import datetime
from sys import stdout

import numpy as np
import tensorflow as tf
from TrainData import TrainData
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from Networks import mobilenet_v2
from Networks.mobilenet_v2 import MobileNetv2_PRN
from Networks.resfcn256_keras import resfcn256_keras


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    train_data_file = args.train_data_file
    learning_rate = args.learning_rate
    epoch_limit = 20
    model_path = args.model_path
    resume_model_path = args.resume_model_path
    resume_model = args.resume_model

    # set tensorflow session GPU usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # set model saving dir
    save_dir = args.checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # initialize TrainData class
    data = TrainData(train_data_file, weight_mask_path='../Data/uv-data/uv_mask_final.png', pre_path='')

    # this appends data from facegen to the dataset.
    # data.training_data_file = '../results/facegen_3dmm_label.txt'# facegen
    # data.read_data(0.8)

    # set model to train on
    if resume_model:
        model = load_model(resume_model_path, custom_objects={'relu6': mobilenet_v2.relu6})
    else:
        if args.model == 'mobilenet':
            model = MobileNetv2_PRN((256, 256, 6), args.alpha_mobilenet)
        elif args.model == 'prnet':
            network = resfcn256_keras()
            model = network.model
        else:
            raise NotImplementedError

        # set model optimizer and compile
        adam = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    print("\n\nstarting training... \n\n")
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fp_log = open("training_logs/log_" + time_now + ".txt", "w")

    # calculate number of iterations on training and validation iterations
    num_train_iterations = int(math.ceil(1.0 * data.num_training_data / batch_size))
    num_validation_iterations = int(math.ceil(1.0 * data.num_validation_data / batch_size))

    # initialize early stopping variables
    epoch_val_losses = [9.999]
    times_not_saved = 0

    for epoch in range(epochs):
        reset_train_metrics = True
        model_saved = False
        np.random.shuffle(data.training_data)
        data.set_augmentation(rotate=True, channel_scale=True, dropout=True)

        # train network on training data
        for iters in range(num_train_iterations):
            batch = data(batch_size, data.training_index, data.num_training_data, data.training_data)
            data.training_index = data.get_updated_index(data.training_index, batch_size, data.num_training_data)
            if iters != 0:
                reset_train_metrics = False
            metrics = model.train_on_batch(x=np.array(batch[0]), y=np.array(batch[1]),
                                           reset_metrics=reset_train_metrics, class_weight=data.weight_mask)
            stdout.flush()
            stdout.write('\riters:%d/%d epoch:%d,loss:%f,accuracy:%f' % (
            iters + 1, num_train_iterations, epoch, metrics[0], metrics[1]))

        # calculate validation loss
        stdout.write('\n\rcalculating validation loss...')
        reset_val_metrics = True
        data.set_augmentation(rotate=False, channel_scale=False, dropout=False)
        for iters in range(num_validation_iterations):
            batch = data(batch_size, data.validation_index, data.num_validation_data, data.validation_data)
            data.validation_index = data.get_updated_index(data.validation_index, batch_size, data.num_validation_data)
            if iters != 0:
                reset_val_metrics = False
            val_metrics = model.test_on_batch(x=np.array(batch[0]), y=np.array(batch[1]),
                                              reset_metrics=reset_val_metrics)

        # early stopping
        if val_metrics[0] < min(epoch_val_losses):
            epoch_val_losses.append(val_metrics[0])
            model.save(model_path)
            model_saved = True
            times_not_saved = 0
        else:
            times_not_saved += 1
            if times_not_saved > epoch_limit:
                break

        stdout.flush()
        stdout.write('\nvalidation loss:%f validation accuracy:%f \n' % (val_metrics[0], val_metrics[1]))
        fp_log.writelines('[%s] epoch:%d learning rate:%f validation loss:%f, validation accuracy:%f, saved:%s\n'
                          % (datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), epoch, learning_rate, val_metrics[0],
                             val_metrics[1], model_saved))

        # cut the learning rate in half every 5/10/3 epochs
        if ((epoch != 0) and (epoch % 5 == 0)):
            learning_rate = learning_rate / 2

    fp_log.close()


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Training Code for Multiview Input Position Map Regression Network')
    par.add_argument('--train_data_file', default='../results/datasetLabel.txt', type=str,
                     help='The training data file')
    par.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate')
    par.add_argument('--epochs', default=200, type=int, help='Total epochs')
    par.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
    par.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='checkpoint/keras_mobilenet_prn.h5', type=str,
                     help='The path of the trained model')
    par.add_argument('--gpu', default='0', type=str, help='The GPU ID')
    par.add_argument('--model', default='mobilenet', type=str,
                     help='Choose which backbone network to train on, mobilenet or prnet')
    par.add_argument('--alpha_mobilenet', default=1.2, type=float, help='Set alpha value for the mobilenet')
    par.add_argument('--resume_model', default=False, type=bool,
                     help='Set if the network should resume training on a pretrained network')
    par.add_argument('--resume_model_path', default='checkpoint/keras_mobilenet_prn_trained.h5', type=str,
                     help='Model path of pretrained network')

    main(par.parse_args())
